import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, cross_val_score
# import pymc as pm
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce
from data_eng_pres import states_with_std, state_readable_with_id, polls
# import arviz as az

################# FUNDAMENTALS MODEL ###################

jobs = pd.read_csv('data/fundamentals/historical_jobs.csv')
spending = pd.read_csv('data/fundamentals/spending.csv')
rpi = pd.read_csv('data/fundamentals/real_personal_income.csv')
manuf = pd.read_csv('data/fundamentals/manufacturing.csv')
inflation = pd.read_csv('data/fundamentals/inflation_cpi.csv')
wages = pd.read_csv('data/fundamentals/avg_real_wages.csv')
gdp = pd.read_csv('data/fundamentals/GDP.csv')
sales = pd.read_csv('data/fundamentals/real_manu_trade_sales.csv')
past_pv = pd.read_csv('data/fundamentals/abramowitz.csv')
cpvi = pd.read_csv('data/fundamentals/cpvi.csv')

consumer_sentiment = pd.read_csv('data/fundamentals/index_of_consumer_sentiment.csv').reset_index().iloc[1:, :3].rename(
    {'level_0':'Month', 'level_1':'Year', 'level_2':'ICS'}, axis='columns'
)
consumer_sentiment['DATE'] = consumer_sentiment['Year'].astype(str) + consumer_sentiment['Month'].map(lambda m: '-0' if int(m) < 10 else '-') + consumer_sentiment['Month'].astype(str) + '-01'
consumer_sentiment = consumer_sentiment[['DATE', 'ICS']]
consumer_sentiment['ICS'] = consumer_sentiment['ICS'].astype(float)

economic_indicators = [jobs, spending, rpi, manuf, inflation, wages, sales, consumer_sentiment, gdp]
fundamentals = reduce(lambda  left,right: pd.merge(left,right,on=['DATE'],
                                            how='inner'), economic_indicators)

def pv_inc(year):
    year_data = past_pv[past_pv['year'] == year]
    inc_party = year_data['inc_party'].values[0]
    if inc_party == 'dem':
        return year_data['dem'].values[0]
    return year_data['rep'].values[0]

def pv_non_inc(year):
    year_data = past_pv[past_pv['year'] == year]
    inc_party = year_data['inc_party'].values[0]
    if inc_party == 'dem':
        return year_data['rep'].values[0]
    return year_data['dem'].values[0]


past_pv['inc_pv'] = past_pv['year'].map(pv_inc)
past_pv['chal_pv'] = past_pv['year'].map(pv_non_inc)

def find_winner(year):
    year_data = past_pv[past_pv['year'] == year].set_index('year')
    if year_data.loc[year, 'dem'] > year_data.loc[year, 'rep']:
        return 'dem'
    return 'rep'

past_pv['winner'] = past_pv['year'].map(find_winner)

date_components = fundamentals['DATE'].str.split('-')
fundamentals['Year'] = date_components.str[0].astype(int)
fundamentals['Month'] = date_components.str[1].astype(int)

def fiscal_quarter(month):
    if month % 3 == 0:
        return month // 3
    return ((month - (month % 3)) // 3) + 1

fundamentals['fiscal_quarter'] = fundamentals['Month'].map(fiscal_quarter)
fundamentals_by_quarter = fundamentals.drop(['DATE', 'Month'], axis=1).groupby(['Year', 'fiscal_quarter']).mean().reset_index()
fundamentals_q1, fundamentals_q2, fundamentals_q3, fundamentals_q4 = tuple([fundamentals_by_quarter[fundamentals_by_quarter['fiscal_quarter'] == k] for k in [1, 2, 3, 4]])
fundamentals_df = fundamentals_q2.merge(past_pv, left_on='Year', right_on='year', how='inner') # Only Q2 economic indicators, at least for now,
# A la Time for Change model

fundamentals_df = fundamentals_df.drop(['inc_start', 'election_date', 'approval_date', 'approval_day', 'approval_day_pulled',
                                       'dem_candidate', 'rep_candidate', 'dem', 'rep', 'total_votes', 'inc_dem', 'inc_rep'], axis=1).replace(
{'dem':0, 'rep':1}
)

# fundamentals_df[['PAYEMS', 'PCE', "W875RX1", 'CMRMTSPL']] = fundamentals_df[['PAYEMS', 'PCE', "W875RX1", 'CMRMTSPL']].apply(lambda x: x / 1000)
fundamentals_df['inc_pv'] = fundamentals_df['inc_pv'] * 100
fundamentals_df['chal_pv'] = fundamentals_df['chal_pv'] * 100

# Now time to build, train, and validate/test our model
X = fundamentals_df[['PAYEMS', 'PCE', 'W875RX1', 'CMRMTSPL', 'CPIAUCSL', 'ICS', 'GDP', 'fte_net_inc_approval']]
y_inc = fundamentals_df['inc_pv']
y_chal = fundamentals_df['chal_pv']

def bootstrap_lasso_once(inc: bool):
    if inc:
        y = y_inc
    else:
        y = y_chal
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model = lm.Lasso(alpha=1, fit_intercept=False, max_iter=5000).fit(X_train, y_train)
    y_pred = model.predict(X)
    return model, y_pred

def bootstrap_lasso_k(k: int, inc: bool):
    predictions = []
    for i in range(k):
        pred = bootstrap_lasso_once(inc)[1]
        predictions.append(pred)
    return np.array(predictions)

def bootstrap_model(k: int, inc: bool):
    models = []
    for i in range(k):
        model = bootstrap_lasso_once(inc)[0]
        models.append(model)
    return models

inc = bootstrap_lasso_k(k=1000, inc=True)
chal = bootstrap_lasso_k(k=1000, inc=False)

pred_inc = sum(inc) / inc.shape[0]
pred_chal = sum(chal) / chal.shape[0]
stdev_inc = np.sqrt(sum((inc - pred_inc)**2) / (inc.shape[0] - 1))
stdev_chal = np.sqrt(sum((chal - pred_chal)**2) / (chal.shape[0] - 1))

# Our 2024 data
data_2024 = fundamentals_q2[fundamentals_q2['Year'] == 2024][['PAYEMS', 'PCE', 'W875RX1', 'CMRMTSPL', 'CPIAUCSL', 'ICS', 'GDP']]
data_2024['fte_net_inc_approval'] = -14.2 # From 538's Joe Biden approval polls

models_inc = bootstrap_model(1000, True)
models_chal = bootstrap_model(1000, False)
inc_predictions = np.array([model.predict(data_2024) for model in models_inc])
chal_predictions = np.array([model.predict(data_2024) for model in models_chal])

expected_temporal_shift = 6.5 # From 538:
# https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=110867585

pred_harris_val = np.mean(inc_predictions)
pred_trump_val = np.mean(chal_predictions)
pred_harris_stdev = np.std(inc_predictions) + expected_temporal_shift
pred_trump_stdev = np.std(chal_predictions) + expected_temporal_shift

# Now to predict our state values from the fundamentals model
cpvi_split = cpvi['CPVI'].str.split('+')
cpvi['partisan_lean'] = cpvi_split.str[0]
cpvi['margin'] = cpvi_split.str[1].astype(int)
dem_pv_2020 = 0.5131
dem_pv_2016 = 0.4818
rep_pv_2020 = 0.4685
rep_pv_2016 = 0.4609
dem_vote = (0.75 * (cpvi['dem_20'] - dem_pv_2020) + 0.25 * (cpvi['dem_16'] - dem_pv_2016))
rep_vote = (0.75 * (cpvi['rep_20'] - rep_pv_2020) + 0.25 * (cpvi['rep_16'] - rep_pv_2016))
cpvi['dem_3pvi'] = dem_vote * 100
cpvi['rep_3pvi'] = rep_vote * 100
cpvi['margin_3pvi'] = (dem_vote - rep_vote) * 100
multiplier = cpvi['partisan_lean'].map(lambda x: -1 if x == 'R' else 1)
cpvi['margin'] = cpvi['margin'] * multiplier
npv_margin = pred_harris_val - pred_trump_val
# By convention, positive = Harris advantage, negative = Trump advantage
projected_state_vote = cpvi['margin_3pvi'].map(lambda x: x + npv_margin)
projected_harris_state_vote = cpvi['dem_3pvi'].map(lambda x: x + pred_harris_val)
projected_trump_state_val = cpvi['rep_3pvi'].map(lambda x: x + pred_trump_val)
cpvi['projected_margin'] = projected_state_vote
cpvi['projected_harris_pct'] = projected_harris_state_vote
cpvi['projected_trump_pct'] = projected_trump_state_val
cpvi['test_margin'] = cpvi['projected_harris_pct'] - cpvi['projected_trump_pct']

state_polling_data = state_readable_with_id.copy()
state_polling_data = state_readable_with_id.merge(polls[['poll_id', 'numeric_grade']], on='poll_id', how='inner')

# def model_one_state(state: str):
#     state_data = cpvi[cpvi['State'] == state]
#     harris_pred = state_data['projected_harris_pct'].values[0]
#     trump_pred = state_data['projected_trump_pct'].values[0]

#     # Get state polling data
#     states_df = states_with_std.copy()
#     state_polling_avg = states_df.reset_index()
#     state_polling_avg_harris = state_polling_avg[state_polling_avg['state'] == state]['Kamala Harris'].values[0]
#     state_polling_avg_trump = state_polling_avg[state_polling_avg['state'] == state]['Donald Trump'].values[0]
#     state_polling_std_harris = state_polling_avg[state_polling_avg['state'] == state]['harris_std'].values[0]
#     state_polling_std_trump = state_polling_avg[state_polling_avg['state'] == state]['trump_std'].values[0]

#     # Now, our Bayesian model
#     harris_model = pm.Model()
#     trump_model = pm.Model()

#     with harris_model:
#         # Priors
#         harris = pm.Normal('harris', mu=harris_pred, sigma=pred_harris_stdev)
#         harris_std = pm.Normal('harris_std_prior', mu=pred_harris_stdev)

#         # Likelihood (sampling distribution) - state polls
#         observed_harris = pm.Normal('harris_polls', mu=harris, sigma=harris_std, observed=pd.to_numeric(state_polling_data[state_polling_data['State'] == state]['Kamala Harris'].to_numpy(), errors='coerce'))
#         # Get posterior
#         # step = pm.Metropolis()
#         trace_harris = pm.sample(1000)# , step=step)

#         # Generate predictions
#         harris_ppc = pm.sample_posterior_predictive(trace_harris)
    
#     with trump_model:
#         # Priors
#         trump = pm.Normal('trump', mu=trump_pred, sigma=pred_trump_stdev)
#         trump_std = pm.Normal('trump_std_prior', mu=pred_trump_stdev)

#         # Likelihood (sampling distribution) - state polls
#         observed_trump = pm.Normal('trump_polls', mu=trump, sigma=trump_std, observed=state_polling_data[state_polling_data['State'] == state]['Donald Trump'].to_numpy())

#         # Get posterior
#         # step = pm.Metropolis()
#         trace_trump = pm.sample(1000)# , step=step) 

#         # Generate predictions
#         trump_ppc = pm.sample_posterior_predictive(trace_trump)
    
#     return trace_harris, trace_trump


# def all_states_model(state_list):
#     dict_pred = {}
#     dict_stderr = {}

#     for state in state_list:
#         model = model_one_state(state)
#         harris_predictions, trump_predictions = model[0], model[1]
# #         harris_means, trump_means = az.summary(harris_predictions, round_to=2)['mean'], az.summary(trump_predictions, round_to=2)['mean']
#         harris_means, trump_means = harris_predictions.posterior['harris'].values.flatten(), trump_predictions.posterior['trump'].values.flatten()
#         harris_winning_prob = np.mean(harris_means > trump_means)
#         dict_pred.update({state: harris_winning_prob})
    
#     return dict_pred



# def main():
# #     pa_model = model_one_state('Pennsylvania')
# #     summary_harris_pa = az.summary(pa_model[0], round_to=2)
# #     summary_trump_pa = az.summary(pa_model[1], round_to=2)
# #     print(type(pa_model[0]))
# #     print(summary_harris_pa)
# #     print(summary_trump_pa)
# #     wi_model = model_one_state('Wisconsin')
# #     summary_harris_wi = az.summary(wi_model[0], round_to=2)
# #     summary_trump_wi = az.summary(wi_model[1], round_to=2)
# #     print(summary_harris_wi)
# #     print(summary_trump_wi)
# #     tx_model = model_one_state('Texas')
# #     summary_harris_tx = az.summary(tx_model[0], round_to=2)
# #     summary_trump_tx = az.summary(tx_model[1], round_to=2)
# #     print(summary_harris_tx)
# #     print(summary_trump_tx)
# #     ca_model = model_one_state('California')
# #     summary_harris_ca = az.summary(ca_model[0], round_to=2)
# #     summary_trump_ca = az.summary(ca_model[1], round_to=2)
# #     print(summary_harris_ca)
# #     print(summary_trump_ca)
#     return all_states_model(['Pennsylvania', 'Wisconsin', 'Michigan', 'Georgia', 'Arizona', 'Minnesota', 'Texas',
#                             'Florida', 'California', 'Ohio'])

# if __name__ == '__main__':
#     print(main())