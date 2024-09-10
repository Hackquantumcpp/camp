import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import datetime
import warnings

warnings.filterwarnings('ignore')

polls = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/president_polls.csv')

# Get dates
dates = polls['end_date'].str.split('/')
polls['end_month'] = dates.str[0].astype(int)
polls['end_day'] = dates.str[1].astype(int)
polls['end_year'] = dates.str[2].astype(int)

# CNN correction
# Based on: https://projects.fivethirtyeight.com/pollster-ratings/
polls_wo_cnn = polls[polls['pollster'] != 'CNN/SSRS']
polls_cnn_ssrs = polls[polls['pollster'] == 'CNN/SSRS']
polls_cnn_ssrs['numeric_grade'] = polls_cnn_ssrs['numeric_grade'].fillna(2.0)
polls = pd.concat([polls_wo_cnn, polls_cnn_ssrs], axis=0)

# Get just recent polls (polls on or after July 1)
rel_polls = polls[(((polls['end_month'] >= 7) & (polls['end_day'] >= 1)) | (polls['end_month'] == 8)) & (polls['end_year'] == 24)
                 & (polls['candidate_name'].isin(['Kamala Harris', 'Donald Trump', 'Robert F. Kennedy', 'Jill Stein', 
                                                  'Cornel West', 'Chase Oliver', 'Claudia de la Cruz']))]

# Only high-quality, nonpartisan polls
# Utilizing an arbitrary cutoff of 2.0
polls_np = rel_polls[rel_polls['numeric_grade'] >= 2.0][rel_polls['partisan'].isna()] # [rel_polls['population'] == 'lv']
polls_for_state_avgs = rel_polls[rel_polls['numeric_grade'] >= 1.5][rel_polls['partisan'].isna()]

# Fill 'NA' values in 'state' with 'National'
polls_np['state'] = polls_np['state'].fillna('National')

# Prepare the dataframe as a time series
polls_np['end_date_TS'] = pd.to_datetime(polls_np['end_date'])
polls_for_state_avgs['end_date_TS'] = pd.to_datetime(polls_for_state_avgs['end_date'])

# Also, highly important SurveyMonkey correction
polls_np = polls_np[polls_np['pollster_rating_name'] != 'SurveyMonkey']

polls_pivot = pd.pivot_table(data=polls_np, values='pct', index=['poll_id', 'state', 'population', 'sample_size', 'end_date_TS', 'pollster_rating_name'], 
                             columns=['candidate_name'], 
                             aggfunc='last', fill_value='NA').reset_index()
polls_pivot_for_states = pd.pivot_table(data=polls_for_state_avgs, values='pct', index=['poll_id', 'state', 'population', 'sample_size', 'end_date_TS', 'pollster_rating_name'], 
                             columns=['candidate_name'], 
                             aggfunc='last', fill_value='NA').reset_index()

def pipeline(data: pd.DataFrame) -> pd.DataFrame:
    # Preferring likely voter samples in polls with multiple samples
    # Find duplicated poll IDs
    # Source for code: 
    # https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
    ids = data['poll_id']
    duplicate_polls = data[ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    unique_polls = data[~ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    dup_polls_lv = duplicate_polls[duplicate_polls['population'] == 'lv']
    df = pd.concat([unique_polls, dup_polls_lv], axis=0)

    ids = df['poll_id']
    duplicate_polls = df[ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    unique_polls = df[~ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    if 'Jill Stein' in duplicate_polls.columns.values:
        dup_polls_3rd_party = duplicate_polls[duplicate_polls['Jill Stein'] != 'NA']
    else:
        dup_polls_3rd_party = duplicate_polls
    df_new = pd.concat([unique_polls, dup_polls_3rd_party], axis=0)
    
    ids = df_new['poll_id']
    duplicate_polls = df_new[ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    unique_polls = df_new[~ids.isin(ids[ids.duplicated()])].sort_values("poll_id")
    dup_polls_rv = duplicate_polls[duplicate_polls['population'] == 'rv']
    df_final = pd.concat([unique_polls, dup_polls_rv], axis=0)
    
    if ((df_final['Kamala Harris'] == 'NA') | (df_final['Donald Trump'] == 'NA')).any():
        df_final = df_final[~((df_final['Kamala Harris'] == 'NA') | (df_final['Donald Trump'] == 'NA'))]
    
    # By convention, positive margins indicate Harris advantage, while negative margins indicate Trump advantage.
    df_final['Margin'] = df_final['Kamala Harris'] - df_final['Donald Trump']
    
    return df_final


polls_pivot = pipeline(polls_pivot)

polls_pivot_with_pollster = polls_pivot.copy()
polls_pivot = polls_pivot.drop(['pollster_rating_name'], axis=1)
polls_with_pollster_readable = polls_pivot_with_pollster.drop(['poll_id'], axis=1).rename({'end_date_TS':'Date',
                                                                                          'pollster_rating_name':'Pollster'}, axis=1)
polls_with_pollster_readable['population'] = polls_with_pollster_readable['population'].str.upper()
polls_with_pollster_readable['Sample'] = polls_with_pollster_readable['sample_size'].astype(int).astype(str) + ' ' + polls_with_pollster_readable['population']
polls_with_pollster_readable = polls_with_pollster_readable.drop(['sample_size', 'population'], axis=1)
nat_readable = polls_with_pollster_readable[polls_with_pollster_readable['state'] == 'National'][['Date', 'Pollster', 'Sample',
                                                                                                 'Kamala Harris', 'Donald Trump',
                                                                                                 'Robert F. Kennedy', 'Jill Stein',
                                                                                                 'Cornel West', 'Chase Oliver']]


state_polls_table = pipeline(polls_pivot_for_states)
state_polls_table = state_polls_table.rename({'end_date_TS':'Date', 'pollster_rating_name':'Pollster'}, axis=1)
state_polls_table['population'] = state_polls_table['population'].str.upper()
state_polls_table['Sample'] = state_polls_table['sample_size'].astype(int).astype(str) + ' ' + state_polls_table['population']
state_polls_table = state_polls_table.drop(['sample_size', 'population'], axis=1)
state_readable = state_polls_table[state_polls_table['state'] != 'National'][['poll_id', 'Date', 'Pollster', 'state', 'Sample',
                                                                                                 'Kamala Harris', 'Donald Trump',
                                                                                                 'Robert F. Kennedy', 'Jill Stein',
                                                                                                 'Cornel West', 'Chase Oliver']].rename({'state':'State'}, axis=1)

polls_pivot = polls_pivot.drop(['population'], axis=1)

# For state polling averages

def produce_raw_df(data: pd.DataFrame) -> pd.DataFrame:
    polls_pivot_raw = data.copy().replace({'NA':0}).drop(['Margin'], axis=1)# .drop(['end_date_TS'], axis=1)
    raw_numbers = polls_pivot_raw[[
        'Chase Oliver', 'Cornel West', 'Donald Trump', 'Jill Stein', 'Kamala Harris', 'Robert F. Kennedy'
    ]].multiply(polls_pivot_raw['sample_size'] * 0.01, axis='index')
    polls_pivot_raw[['Chase Oliver', 'Cornel West', 'Donald Trump', 'Jill Stein', 'Kamala Harris',
                     'Robert F. Kennedy']] = raw_numbers
    
    return polls_pivot_raw

polls_pivot_raw = produce_raw_df(polls_pivot)
polls_pivot_raw_full = polls_pivot_raw.copy()
polls_pivot_raw = polls_pivot_raw[polls_pivot_raw['end_date_TS'] >= pd.to_datetime('2024-07-24')]

# By convention, positive margins indicate Harris advantage, while negative margins indicate Trump advantage.
# Utilizing weighted averages taking sample size into account
# states = polls_pivot[['state', 'Margin']].groupby(['state']).agg({'Margin':'mean'})
states = polls_pivot_raw.drop(['poll_id'], axis=1).groupby(['state'])[['Kamala Harris', 'Donald Trump', 'sample_size']].sum()
states[['Kamala Harris', 'Donald Trump']] = states[['Kamala Harris', 'Donald Trump']].multiply(1 / states['sample_size'],
                                                                                               axis='index')
states['Margin'] = (states['Kamala Harris'] - states['Donald Trump']) * 100
states = states.drop(['sample_size', 'Kamala Harris', 'Donald Trump'], axis=1)

def margin_rating(margin):
    abs_margin = abs(margin)
    if abs_margin >= 15:
        rating = 'Solid'
    elif abs_margin < 15 and abs_margin >= 5:
        rating = 'Likely'
    elif abs_margin < 5 and abs_margin >= 2:
        rating = 'Lean'
    else:
        rating = 'Tossup'
    
    if rating != 'Tossup':
        if margin < 0:
            direc = ' R'
        else:
            direc =  ' D'
    else:
        direc = ''
    
    return rating + direc

def margin_with_party(margin):
    if margin < 0:
        direc = 'R'
    else:
        direc = 'D'
    return direc + '+' + f'{abs(margin):.1f}'

states['Rating'] = states['Margin'].map(margin_rating)
states['Label'] = states['Margin'].map(margin_with_party)
states_preproc = states.copy()

# NEW State Polling Averages!

def state_avgs_pipeline(state: str):
    state_race = polls_for_state_avgs[polls_for_state_avgs['state'] == state]# [senate['party'].isin(['DEM', 'REP'])]
    state_pivot = pd.pivot_table(data=state_race, values='pct', index=['poll_id', 'display_name', 'state', 'end_date', 'sample_size',
                                                               'population', 'numeric_grade'], columns=['candidate_name'],
                                                              aggfunc='last', fill_value='NA').reset_index()
    state_pivot['end_date'] = pd.to_datetime(state_pivot['end_date'])
    state_pivot = state_pivot[state_pivot['end_date'] >= pd.to_datetime('2024-07-24')]
    state_pivot = pipeline(state_pivot)
    
    total_num_polls = state_pivot.shape[0]

    # Sample size weights
    total_sample_size = np.sum(state_pivot['sample_size'])
    state_pivot['sample_size_weights'] = (state_pivot['sample_size'].map(np.sqrt) / np.sqrt(np.median(state_pivot['sample_size'])))
    state_pivot['sample_size_weights'] /= np.sum(state_pivot['sample_size_weights'])
    
    # Time weights
    # Variation of the equation used here: https://polls.votehub.us/
    today = datetime.datetime.today()
    delta = (today - state_pivot['end_date']).apply(lambda x: x.days)
    linear_weights = (1 - delta/((today - state_pivot['end_date'].min()).days + 1))
    exp_weights = 0.4**(delta/((today - state_pivot['end_date'].min()).days + 1))
    state_pivot['time_weights'] =  0.3 * linear_weights + 0.7 * exp_weights
    state_pivot['time_weights'] /= np.sum(state_pivot['time_weights'])
    
    # Quality weights
    min_quality = 1.5
    rel_quality = state_pivot['numeric_grade'] - min_quality
    def quality_weight(rel_qual):
        if rel_qual < -0.2:
            return 0.01
        elif rel_qual < 0:
            return 0.02
        return (0.05 + (0.95/(3-min_quality)) * rel_qual)
    state_pivot['quality_weights'] = rel_quality.map(quality_weight)
    state_pivot['quality_weights'] /= np.sum(state_pivot['quality_weights'])
    
    # Population weights
    def population_weight(population):
        if population == 'a':
            return 0.6
        return 1
    state_pivot['population_weights'] = state_pivot['population'].map(population_weight)
    state_pivot['population_weights'] /= np.sum(state_pivot['population_weights'])

    # Gather the weights together
    state_pivot['total_weights'] = state_pivot['sample_size_weights'] * state_pivot['time_weights'] * state_pivot['quality_weights'] * state_pivot['population_weights']
    state_pivot['total_weights'] /= np.sum(state_pivot['total_weights']) # Normalization step
    
    return state_pivot
def all_state_polls_with_weights(state_list):
    df = state_avgs_pipeline(state_list[0]).replace({'NA':0})
    for state in state_list[1:]:
        pipelined_df = state_avgs_pipeline(state).replace({'NA':0})
        df = pd.concat([df, pipelined_df], axis=0)
    return df

def get_state_averages(state_list):
    dem_avgs = []
    rep_avgs = []
    for state in state_list:
        pipelined_df = state_avgs_pipeline(state).replace({'NA':0})
        dem_avg = np.sum(pipelined_df['Kamala Harris'] * pipelined_df['total_weights'])
        rep_avg = np.sum(pipelined_df['Donald Trump'] * pipelined_df['total_weights'])
        dem_avgs.append(dem_avg)
        rep_avgs.append(rep_avg)
    
    return pd.DataFrame({'state':state_list.tolist(), 'Kamala Harris':dem_avgs, 'Donald Trump':rep_avgs})
states_preproc_reset = states_preproc.reset_index()
state_avgs_experimental = get_state_averages(states_preproc_reset[states_preproc_reset['state'] != 'National']['state'].values)
# By convention, positive margins indicate Harris advantage, while negative margins indicate Trump advantage.
state_avgs_experimental['Margin'] = state_avgs_experimental['Kamala Harris'] - state_avgs_experimental['Donald Trump']
state_avgs_experimental['Rating'] = state_avgs_experimental['Margin'].map(margin_rating)
state_avgs_experimental['Label'] = state_avgs_experimental['Margin'].map(margin_with_party)
states_preproc = state_avgs_experimental.copy()
states = state_avgs_experimental.copy()

states_ec = pd.read_csv('data/other/electoral-votes-by-state-2024.csv')
states_abb = pd.read_csv('data/other/Electoral_College.csv').drop(['Electoral_College_Votes'], axis=1)


states = states.reset_index().merge(states_ec, on='state').merge(states_abb, left_on='state', right_on='Full_State')
states['margin_for_choropleth'] = states['Margin'].map(lambda x: min(x, 20))

weights = all_state_polls_with_weights(states_preproc[states_preproc['state'] != 'National']['state'].values)[['poll_id', 'total_weights']]
state_readable = state_readable.merge(weights, on='poll_id').rename({'total_weights':'Weight in State Polling Average'}, axis=1)
state_readable['Weight in State Polling Average'] = state_readable['Weight in State Polling Average'].apply(lambda x: float(f'{x:.5f}'))
state_readable = state_readable.drop(['poll_id'], axis=1)

# states_preproc = states_preproc.reset_index()
competitive = states_preproc[states_preproc['state'].isin(['Arizona', 'Georgia', 'Pennsylvania', 
                                                           'Michigan', 'Wisconsin', 'North Carolina', 'Minnesota',
                                          'Nevada', 'Texas', 'Florida', 'New Hampshire', 'Maine', 'Maine CD-2',
                                                          'Nebraska CD-2', 'Virginia', 'New Mexico', 'Ohio'])]
competitive = competitive.sort_values(by=['Margin'], ascending=False).merge(states_ec, on='state')
leader = lambda x: 'Republicans' if x < 0 else 'Democrats'
competitive['Leader'] = competitive['Margin'].map(leader)

harris_polled_ev = 191 + competitive[competitive['Leader'] == 'Democrats']['ElectoralVotes'].sum()
trump_polled_ev = 131 + competitive[competitive['Leader'] == 'Republicans']['ElectoralVotes'].sum()
def find_tipping_point():
    if harris_polled_ev > trump_polled_ev:
        df = competitive[competitive['Leader'] == 'Democrats'].copy()
        curr = harris_polled_ev
    elif trump_polled_ev > harris_polled_ev:
        df = competitive[competitive['Leader'] == 'Republicans'].copy()
        curr = trump_polled_ev
    else:
        return 'Tie'
    df['Margin'] = df['Margin'].map(abs)
    df = df.sort_values(by=['Margin'], ignore_index=True)
    new_df = df.copy()
    while curr - new_df.iloc[0, 6] > 269:
        curr -= new_df.iloc[0, 6]
        new_df = new_df[1:]
    return new_df.iloc[0, 0]

tp_state = find_tipping_point()
tp_margin = states[states['state'] == tp_state]['Margin'].values[0]



# National polling averages over time
polling_avgs_by_day = polls_pivot_raw_full.drop(['poll_id'], axis=1).groupby(['end_date_TS', 'state']).sum()
polling_avgs_by_day[['Chase Oliver', 'Cornel West', 'Donald Trump', 'Jill Stein',
                    'Kamala Harris', 'Robert F. Kennedy']] = polling_avgs_by_day[['Chase Oliver', 'Cornel West', 
                                                                                  'Donald Trump', 
                                                                                  'Jill Stein', 'Kamala Harris', 
                                                                                  'Robert F. Kennedy']].multiply(100 / polling_avgs_by_day['sample_size'], 
                                                                                                                 axis='index')

polling_avgs = polling_avgs_by_day.reset_index()

national_polling_avgs = polling_avgs[polling_avgs['state'] == 'National']

from statsmodels.nonparametric.smoothers_lowess import lowess

# polls_ts = pd.pivot_table(data=polls_np, values='pct', index=['poll_id', 'end_date_TS', 'state'], columns=['candidate_name'],
#                          aggfunc='first', fill_value='NA')

# Utilizing the polls_pivot dataframe, which has been feature engineered to prefer likely voter samples in polls with
# multiple samples to registered voter and all adult samples.
polls_ts = polls_pivot.copy()
nat_polls_ts = polls_ts[polls_ts['state'] == 'National']
# # No NA values
nat_polls_ts = nat_polls_ts[~((nat_polls_ts['Kamala Harris'] == 'NA') | (nat_polls_ts['Donald Trump'] == 'NA'))]
nat_polls_ts['Kamala Harris'] = nat_polls_ts['Kamala Harris'].map(float)
harris_nat = nat_polls_ts['Kamala Harris'].to_numpy()
trump_nat = nat_polls_ts['Donald Trump'].to_numpy()
dates = nat_polls_ts['end_date_TS'].to_numpy()
trump_lowess = lowess(trump_nat, dates, frac=0.3, it=5, return_sorted=True)
harris_lowess = lowess(harris_nat, dates, frac=0.3, it=5, return_sorted=True)
dates_lowess = pd.to_datetime(trump_lowess[:, 0])
nat_polls_ts_ind = nat_polls_ts.groupby(['end_date_TS'])[['Kamala Harris', 'Donald Trump']].mean()
# Calculate exponential weighted moving average
harris_ewma = nat_polls_ts_ind['Kamala Harris'].ewm(alpha=0.1).mean()
trump_ewma = nat_polls_ts_ind['Donald Trump'].ewm(alpha=0.1).mean()
harris_ewma_std = nat_polls_ts_ind['Kamala Harris'].ewm(alpha=0.1).std()
trump_ewma_std = nat_polls_ts_ind['Donald Trump'].ewm(alpha=0.1).std()
harris_lowess_df = pd.DataFrame(harris_lowess)
harris_lowess_df[0] = harris_lowess_df[0].map(pd.to_datetime)
harris_lowess_df = harris_lowess_df.set_index(0)
harris_curves = pd.DataFrame(harris_ewma).join(harris_lowess_df, how='outer').rename({'Kamala Harris':'Harris EWMA',
                                                                                     1:'Harris Lowess'}, axis=1).reset_index()
harris_curves = harris_curves.groupby(['index']).mean()
# harris_curves
trump_lowess_df = pd.DataFrame(trump_lowess)
trump_lowess_df[0] = trump_lowess_df[0].map(pd.to_datetime)
trump_lowess_df = trump_lowess_df.set_index(0)
trump_curves = pd.DataFrame(trump_ewma).join(trump_lowess_df, how='outer').rename({'Donald Trump':'Trump EWMA',
                                                                                     1:'Trump Lowess'}, axis=1).reset_index()
trump_curves = trump_curves.groupby(['index']).mean()
# trump_curves
def polling_average(lowess, ewma, mixing_param=0):
    """
    Calculate weighted average of LOWESS and EWMA curve.
    :param mixing_param: Mixing parameter. The mixing parameter is added to the LOWESS weight and subtracted from
    the EWMA weight.
    """
    return (0.5 + mixing_param) * lowess + (0.5 - mixing_param) * ewma
# Just use only LOWESS for now, I am tired
harris_ts = polling_average(harris_curves['Harris Lowess'], harris_curves['Harris EWMA'], mixing_param=0.5)
trump_ts = polling_average(trump_curves['Trump Lowess'], trump_curves['Trump EWMA'], mixing_param=0.5)
harris_curves['polling_avg'] = harris_ts
trump_curves['polling_avg'] = trump_ts
# dates_ts = np.union1d(dates_lowess, )
# harris_ts, trump_ts

# Interpolation

dates_range = pd.date_range(start=harris_curves.index.min(), end=harris_curves.index.max(),freq='d',
                           inclusive='both')
dates_range_num = pd.to_numeric(dates_range)
harris_interp = scipy.interpolate.interp1d(pd.to_numeric(harris_curves.index.to_numpy()), 
                                           harris_curves['polling_avg'].to_numpy())(dates_range_num)
trump_interp = scipy.interpolate.interp1d(pd.to_numeric(trump_curves.index.to_numpy()), 
                                           trump_curves['polling_avg'].to_numpy())(dates_range_num)
harris_trump_data_interp = pd.DataFrame([harris_interp, trump_interp, dates_range]).T.rename(columns={0:'Kamala Harris',
                                                                                                   1: 'Donald Trump',
                                                                                                   2:'Date'})

# Calculating margin of error (95% confidence intervals)
# Code from: https://james-brennan.github.io/posts/lowess_conf/
def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), 50, replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = lowess(y_s,x_s, frac=0.3, it=5,
                     return_sorted = False)
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, 
                                        fill_value='extrapolate')(xgrid)
    return y_grid
dates_num = pd.to_numeric(dates)
xgrid = np.linspace(dates_num.min(), dates_num.max())
dates_grid = pd.to_datetime(xgrid)
K = 100
harris_smooths = np.stack([smooth(dates_num, harris_nat, dates_range_num) for k in range(K)]).T
trump_smooths = np.stack([smooth(dates_num, trump_nat, dates_range_num) for k in range(K)]).T
mean_h = np.nanmean(harris_smooths, axis=1)
stderr_h = scipy.stats.sem(harris_smooths, axis=1)
stderr_h = np.nanstd(harris_smooths, axis=1, ddof=0)
mean_t = np.nanmean(trump_smooths, axis=1)
stderr_t = scipy.stats.sem(trump_smooths, axis=1)
stderr_t = np.nanstd(trump_smooths, axis=1, ddof=0)

###############

harris_trump_data = harris_curves.join(trump_curves, how='outer', lsuffix='_harris', rsuffix='_trump')
harris_trump_data = harris_trump_data[['polling_avg_harris', 'polling_avg_trump']].reset_index().rename(
    {'polling_avg_harris':'Kamala Harris',
     'polling_avg_trump':'Donald Trump',
    'index':'Date'}, axis=1)

nat_polls_ts_readable = nat_polls_ts.rename({'end_date_TS':'Date'}, axis=1)



# Plots
# fig_line = px.line(data_frame=harris_trump_data, x='Date', y=['Kamala Harris', 'Donald Trump'], 
#                    title='Harris vs. Trump National Polling', markers=False)

fig_line = px.line(data_frame=harris_trump_data_interp, x='Date', y=['Kamala Harris', 'Donald Trump'], 
                   title='Harris vs. Trump National Polling', markers=False)
# fig_line.show()

fig_scatter = px.scatter(data_frame=nat_polls_ts_readable, x='Date', y=['Kamala Harris', 'Donald Trump'], opacity=0.5)
# fig_scatter.show()

fig_harris_CI = go.Figure([
    go.Scatter(
        name='Harris CI Upper Bound',
        x = dates_range,
        y = mean_h + 1.96*stderr_h,
        mode='lines',
        marker=dict(color='#8972fc'),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    go.Scatter(
        name='Harris CI Lower Bound',
        x = dates_range,
        y = mean_h - 1.96*stderr_h,
        mode='lines',
        marker=dict(color='#8972fc'),
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(137, 114, 252, 0.15)',
        showlegend=False,
        hoverinfo='skip'
    )
    
])

fig_trump_CI = go.Figure([
    go.Scatter(
        name='Trump CI Upper Bound',
        x = dates_range,
        y = mean_t + 1.96*stderr_t,
        mode='lines',
        marker=dict(color='#fc7472'),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    go.Scatter(
        name='Trump CI Lower Bound',
        x = dates_range,
        y = mean_t - 1.96*stderr_t,
        mode='lines',
        marker=dict(color='#fc7472'),
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(252, 116, 114, 0.15)',
        showlegend=False,
        hoverinfo='skip'
    )
    
])

fig = go.Figure(data=fig_line.data + fig_scatter.data + fig_harris_CI.data + fig_trump_CI.data)

fig.update_layout(
    title='Harris v. Trump Polling for the 2024 Presidential Election',
    xaxis_title = 'Date',
    yaxis_title = 'Polled Vote %',
    legend_title = 'Legend',
)

fig.add_vline(x=datetime.datetime.strptime("2024-07-21", "%Y-%m-%d").timestamp() * 1000, line_dash='dot', 
              annotation_text='Biden drops out', annotation_position='top right')


fig_states = px.choropleth(data_frame=states.reset_index(), locations='Abb_State', locationmode='USA-states', 
                           color='margin_for_choropleth',
                          color_continuous_scale='RdBu', range_color=[-20, 20], hover_name='state', 
                          hover_data={'Abb_State':False, 'Rating':True, 'Margin':False, 'Label':True, 
                                      'margin_for_choropleth':False},
                          labels={'Label':'Average Margin'}, width=1400, height=1000)
fig_states.update_layout(
    title_text = '2024 US Presidential Election State Polling Averages',
    geo_scope='usa', # limit map scope to USA
)

fig_states.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-20, -10, 0, 10, 20],
    ticktext=['>R+20', 'R+10', 'EVEN', 'D+10', '>D+20']
))
states = states.rename({'Margin':'Average Polling Margin'}, axis=1)

fig_comp = px.bar(data_frame=competitive, x='Margin', y='state', color='Leader')

fig_comp.update_layout(
    title='Margins in Competitive States',
    yaxis_title='State'
)

# EC Bias
avg_lowess_diff = harris_trump_data_interp['Kamala Harris'].to_numpy()[-1] - harris_trump_data_interp['Donald Trump'].to_numpy()[-1]
nat_diff = ('Harris' if avg_lowess_diff > 0 else 'Trump') + '+' + f'{abs(avg_lowess_diff):.2f}%'
ec_bias = tp_margin - avg_lowess_diff
ec_bias_pres = ('D' if ec_bias > 0 else 'R') + '+' + f'{abs(ec_bias):.2f}%'
