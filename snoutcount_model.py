import numpy as np
import pandas as pd
# import sklearn.linear_model as lm
# from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objects as go
# from functools import reduce
import scipy
from statsmodels.stats.moment_helpers import corr2cov
# from statsmodels.stats.correlation_tools import cov_nearest, corr_nearest, corr_clipped
import warnings
import datetime
from pathlib import Path

from data_eng_pres import states_with_std_all, state_readable_with_id, polls, states_ec, nat_diff, harris_trump_data_interp, states_abb, margin_rating, margin_with_party
from fundamentals_model_output import pred_harris_stdev, pred_trump_stdev

corr_matrix = pd.read_csv('data/fundamentals/state_correlation_matrix.csv')
cpvi = pd.read_csv('data/fundamentals/cpvi_modified.csv')
polls_movement_df = pd.read_csv('data/fundamentals/polls_movement.csv') # Hand typed, data from:
# https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753
# Only includes polls movement up to 35 days
election_day = datetime.date(2024, 11, 5)
today = datetime.date.today()
days_left = (election_day - today).days

expected_polling_shift = polls_movement_df[polls_movement_df['days_before_nov5'] == days_left]['movement']
expected_polling_error = 4 + expected_polling_shift
weights = state_readable_with_id['Weight in State Polling Average'].to_numpy()

states_ec_dict = states_ec.set_index('state').to_dict('index')
state_list = states_ec['state'].to_numpy()
full_state_list = state_list[state_list != 'United States']

def chance_for_state(state: str):
    """
    Return chance Kamala Harris wins a state, based on frequentist sampling of a distribution generated from
    stats determined solely by polls.
    """
    state_data = states_with_std_all[states_with_std_all['state'] == state]
    harris, trump = state_data['Kamala Harris'].values[0], state_data['Donald Trump'].values[0]
    harris_std, trump_std = state_data['harris_std'].values[0], state_data['trump_std'].values[0]
    
    harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=harris_std + expected_polling_error, 
                                    size=10001)
    trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=trump_std + expected_polling_error, 
                                   size=10001)
    
    return np.count_nonzero(harris_dist > trump_dist) / harris_dist.shape[0]

def chance_for_unpolled_state(state: str):
    state_data = cpvi[cpvi['State'] == state]
    harris_nat, trump_nat = harris_trump_data_interp['Kamala Harris'].to_numpy()[-1], harris_trump_data_interp['Donald Trump'].to_numpy()[-1]
    harris, trump = harris_nat + state_data['dem_3pvi'].values[0], trump_nat + state_data['rep_3pvi'].values[0]
    
    harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=expected_polling_error, 
                                    size=10001)
    trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=expected_polling_error, 
                                   size=10001)
    
    return np.count_nonzero(harris_dist > trump_dist) / harris_dist.shape[0]

def margins_for_unpolled_states(state_list):
    harris_nat, trump_nat = harris_trump_data_interp['Kamala Harris'].to_numpy()[-1], harris_trump_data_interp['Donald Trump'].to_numpy()[-1]
    cpvi_si = cpvi.set_index(['State'])
    harris, trump = cpvi_si['dem_3pvi'].map(lambda x: x + harris_nat), cpvi_si['rep_3pvi'].map(lambda x: x + trump_nat)
#     margins = harris - trump # Convention: positive = Harris, negative = Trump
#     margins = margins[~margins.index.isin(states_with_std_all['state'].values)]
    harris_std, trump_std = np.full(shape=harris.shape, fill_value=expected_polling_error), np.full(shape=harris.shape, fill_value=expected_polling_error)
    df = pd.concat([harris, trump], axis=1)
    df['harris_std'], df['trump_std'] = harris_std, trump_std
    df['margin'], df['margin_std'] = df['dem_3pvi'] - df['rep_3pvi'], np.full(shape=harris.shape, fill_value=expected_polling_error)
    df = df.reset_index().rename({'dem_3pvi':'Kamala Harris', 'rep_3pvi':'Donald Trump', 'State':'state'}, axis=1)
    df = df[~df['state'].isin(states_with_std_all['state'].values)]
    return df

def all_state_polled_margins():
    unpolled = margins_for_unpolled_states(full_state_list)
    df = pd.concat([states_with_std_all, unpolled], axis=0)
    
    # Convention: positive = Harris advantage, negative = Trump advantage
    # df['margin'] = df['Kamala Harris'] - df['Donald Trump']
    
    df = df.sort_values(by=['state'])
    # df['margin_std'] = np.sqrt(df['harris_std']**2 + df['trump_std']**2)
    return df

############################################################################################

def simple_chances(return_samples=False):
    # For CDs that get their own EVs
    margins_df = all_state_polled_margins()
    margins_df = margins_df[margins_df['state'].isin(['Maine CD-1', 'Maine CD-2', 'Nebraska CD-1', 'Nebraska CD-2', 'Nebraska CD-3'])]
    chances = {}
    samples = {}
    for state in margins_df['state']:
        state_data = margins_df[margins_df['state'] == state]
        harris, trump = state_data['Kamala Harris'].values[0], state_data['Donald Trump'].values[0]
        harris_std, trump_std = state_data['harris_std'].values[0], state_data['trump_std'].values[0]
        harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=(harris_std + expected_polling_error if harris_std != 2 else harris_std), 
                                    size=10001)
        trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=(trump_std + expected_polling_error if trump_std != 2 else trump_std), 
                                   size=10001)
        chances.update({state: np.count_nonzero(harris_dist > trump_dist) / harris_dist.shape[0]})
        samples.update({state: harris_dist - trump_dist})
    df = pd.DataFrame(pd.Series(chances))
    df = pd.concat([margins_df[['state', 'margin']].set_index(['state']), df], axis=1).rename({0: 'chance'}, axis=1)
    if return_samples:
        return df, samples
    return df

# Big thanks to the Alan Turing Institute for the is_positive_definite and nearest_positive_definite (for them, isPD and NPD) functions!
# Taken from BOCPDMS source code, which is licensed with MIT license
def is_positive_definite(B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to input
    
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def adjust_margins(k=10000, method='cholesky', return_chances=False, return_samples=False):
    """
    Give adjusted margins (and, if return_chances is True, Harris win chances) for each state using state correlation matrix.
    k: number of simulations to run.
    """
    # Get state margins
    margins_df = all_state_polled_margins()
    margins_df = margins_df[~margins_df['state'].isin(['Maine CD-1', 'Maine CD-2', 'Nebraska CD-1', 'Nebraska CD-2', 'Nebraska CD-3'])]
    margins = margins_df['margin'].values
    margin_stdevs = margins_df['margin_std'].values
    
    # Calculate covariance matrix
    cov_matrix = corr2cov(corr_matrix, margin_stdevs)
    
    if method == 'cholesky':
        # Calculate Cholesky decomposition of covariance matrix
        try:
            L = scipy.linalg.cholesky(cov_matrix, lower=True)
        except scipy.linalg.LinAlgError:
            epsilon = 1e-6
            try:
                L = scipy.linalg.cholesky(cov_matrix + (np.eye(cov_matrix.shape[0]) * epsilon), lower=True)
            except scipy.linalg.LinAlgError or ValueError:
                # cov_matrix_pdm = corr2cov(corr_clipped(corr_matrix), margin_stdevs)
                try:
                    cov_matrix_pdm = nearest_positive_definite(cov_matrix)
                    L = scipy.linalg.cholesky(cov_matrix_pdm, lower=True)
                except scipy.linalg.LinAlgError:
                    warnings.warn('Failed to find nearest positive definite matrix. Utilizing absolute values of eigenvalues instead.')
                    eigenvals, eigenvecs = scipy.linalg.eigh(cov_matrix)
                    L = np.dot(eigenvecs, np.diag(np.sqrt(abs(eigenvals))))
                    if np.isnan(L).any() and np.isinf(L).any():
                        raise ValueError('L has both NaN and inf values.')
                    if np.isnan(L).any():
                        raise ValueError('L has NaN values.')
                    if np.isinf(L).any():
                        raise ValueError('L has inf values.')

        # Generated correlated noise
        noise = np.random.normal(0, margin_stdevs, (k, margin_stdevs.shape[0])) # Uncorrelated
        # noise = np.random.standard_t(df=3, size=(k, margin_stdevs.shape[0]))
        correlated_noise = noise @ L.T
        
        # Compute adjusted margins
        adjusted_margins = np.mean(margins + correlated_noise, axis=0)
        adjusted_std = np.std(margins + correlated_noise, axis=0)
    
    elif method == 'multivariate_normal_sampling':
        # Randomly sample from multivariate normal distribution
        adjusted_margins_samples = scipy.stats.multivariate_normal.rvs(margins, cov=cov_matrix, size=k)
        adjusted_margins = np.mean(adjusted_margins_samples, axis=0)
        adjusted_std = np.std(adjusted_margins_samples, axis=0)
        
    else:
        raise ValueError("Please input either 'cholesky' or 'multivariate_normal_sampling' as the method parameter.")
    
    adjusted_margins = pd.Series(adjusted_margins)
    # adjusted_std = pd.Series(adjusted_std)
    states = margins_df.reset_index()['state']
    adjusted_margins_df = pd.concat([states, adjusted_margins], axis=1).set_index(['state']).rename({0: 'margin'}, axis=1)
    adjusted_margins = adjusted_margins_df['margin']
    
    if return_chances:
        chances = {}
        samples_dict = {}
        # print(np.column_stack((adjusted_margins.values, margin_stdevs, states.values)))
#         for margin in np.column_stack((adjusted_margins.values, margin_stdevs, states.values)):
#             margin_std_used = (margin[1] + 2) if margin[1] != 2 else (margin[1])
#             samples = scipy.stats.t.rvs(df=5, loc=margin[0], scale=margin_std_used, size=10001)
#             chance = np.mean(samples > 0)
#             chances.update({margin[2]: chance})
#             samples_dict.update({margin[2]: samples})
        try:
            adjusted_margins_samples = scipy.stats.multivariate_normal.rvs(adjusted_margins.values, cov=cov_matrix, size=k+1)
        except ValueError:
            adjusted_margins_samples = scipy.stats.multivariate_normal.rvs(adjusted_margins.values, cov=nearest_positive_definite(cov_matrix), size=k+1)
#        dof = 5
#        adjusted_margins_samples = scipy.stats.multivariate_t(df=5, shape=nearest_positive_definite(cov_matrix), loc=np.zeros((states.shape[0], k+1))) 
        # print(adjusted_margins_samples)
#         print(adjusted_margins_samples.shape)
        for i in range(adjusted_margins_samples.shape[1]):
            chances.update({full_state_list[i]: np.mean(adjusted_margins_samples[:, i] > 0)})
            samples_dict.update({full_state_list[i]: adjusted_margins_samples[:, i]})
        adjusted_margins_df['chance'] = chances
        if return_samples:
            return adjusted_margins_df, samples_dict
        return adjusted_margins_df
    
    return adjusted_margins

chances_df, samples = adjust_margins(return_chances=True, return_samples=True)
cd_chances, cd_samples = simple_chances(return_samples=True)
chances_df = pd.concat([chances_df, simple_chances()], axis=0)
samples.update(cd_samples)

def ev_pred():
    """
    Prediction of number of EVs that Harris will get, from a poll-based model.
    """
    harris_ec = np.zeros(10001)
    trump_ec = np.zeros(10001)
    def harris_winner(margin):
        return 1 if margin > 0 else 0
    def trump_winner(margin):
        return 0 if margin > 0 else 1
    # def evs(margin, state):
    #     return (states_ec_dict[state]['ElectoralVotes'] * winner(margin))
    winner_dict = samples.copy()
    for state in full_state_list:
        state_ec = states_ec_dict[state]['ElectoralVotes']
        harris_winning = np.vectorize(harris_winner)(winner_dict[state])
        trump_winning = np.vectorize(trump_winner)(winner_dict[state])
        harris_ev = harris_winning * state_ec
        trump_ev = trump_winning * state_ec
        harris_ec += harris_ev
        trump_ec += trump_ev
    return {'harris': np.mean(harris_ec > trump_ec), 'trump': np.mean(trump_ec > harris_ec), 'tie': np.mean(harris_ec == trump_ec)}


###############################

def polls_dist(state: str):
    """
    Return chance Kamala Harris wins a state, based on frequentist sampling of a distribution generated from
    stats determined solely by polls.
    """
    state_data = states_with_std_all[states_with_std_all['state'] == state]
    harris, trump = state_data['Kamala Harris'].values[0], state_data['Donald Trump'].values[0]
    harris_std, trump_std = state_data['harris_std'].values[0], state_data['trump_std'].values[0]
    
    harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=harris_std + expected_polling_error, 
                                    size=10001)
    trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=trump_std + expected_polling_error, 
                                   size=10001)
    
    return harris_dist, trump_dist

def fundamentals_based_chance(state: str, return_samples=False):
    """
    Return chance Kamala Harris wins a state, based on frequentist sampling of a distribution generated from
    stats determined solely by fundamentals.
    """
    state_data = cpvi[cpvi['State'] == state]
    harris, trump = state_data['projected_harris_pct'].values[0], state_data['projected_trump_pct'].values[0]
    div_factor = 4
    harris_std, trump_std = pred_harris_stdev / div_factor, pred_trump_stdev / div_factor
    # harris_std, trump_std = 15, 15
    
    harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=harris_std, 
                                    size=10001)
    trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=trump_std, 
                                   size=10001)
    if return_samples:
        return harris_dist - trump_dist
    return np.count_nonzero(harris_dist > trump_dist) / harris_dist.shape[0]

def all_chances(state_list):
    chances = pd.DataFrame(columns=['chance'], index=['state'])
    for state in state_list:
        chance = chance_for_state(state)
        state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
        chances = pd.concat([chances, state_chance_df], axis=0)
    return chances[1:]

def all_unpolled_chances(state_list):
    chances = pd.DataFrame(columns=['chance'], index=['state'])
    for state in state_list:
        chance = chance_for_unpolled_state(state)
        state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
        chances = pd.concat([chances, state_chance_df], axis=0)
    return chances[1:]

def fundamentals_pred(state_list, return_samples=False):
    chances = pd.DataFrame(columns=['chance'], index=['state'])
    samples = {}
    for state in state_list:
        chance = fundamentals_based_chance(state, return_samples=return_samples)
        if return_samples:
            samples.update({state: chance})
            continue
        state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
        chances = pd.concat([chances, state_chance_df], axis=0)
    if return_samples:
        return samples
    return chances[1:]

def fundamentals_ev_pred_old():
    harris_ec = np.zeros(10001)
    trump_ec = np.zeros(10001)
    # state_chances = {}
    state_chances = pd.DataFrame(columns=['chance'], index=['state'])
    for state in full_state_list:
        state_data = cpvi[cpvi['State'] == state]
        harris, trump = state_data['projected_harris_pct'].values[0], state_data['projected_trump_pct'].values[0]
        div_factor = 4
        harris_std, trump_std = pred_harris_stdev / div_factor, pred_trump_stdev / div_factor
        harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=harris_std, 
                                        size=10001)
        trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=trump_std, 
                                       size=10001)
        chance = np.mean(harris_dist > trump_dist)
        state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
        state_chances = pd.concat([state_chances, state_chance_df], axis=0)
        def ec_adder(winning: bool):
            return (states_ec_dict[state]['ElectoralVotes'] if winning else 0)
        harris_ec += np.array([ec_adder(v) for v in (harris_dist > trump_dist)])
        trump_ec += np.array([ec_adder(v) for v in (trump_dist > harris_dist)])
    chance = np.count_nonzero(harris_ec > trump_ec) / harris_ec.shape[0]
    pred = np.median(harris_ec)
    return {'harris': np.mean(harris_ec > trump_ec), 'trump': np.mean(trump_ec > harris_ec), 'tie': np.mean(harris_ec == trump_ec)}, state_chances[1:]

def fundamentals_ev_pred():
    harris_ec = np.zeros(10001)
    trump_ec = np.zeros(10001)
    # state_chances = {}
    state_chances = pd.DataFrame(columns=['chance'], index=['state'])
    harris = cpvi[~cpvi['State'].str.contains('CD')]['projected_harris_pct'].values
    trump = cpvi[~cpvi['State'].str.contains('CD')]['projected_trump_pct'].values
    div_factor = 4
    cov_matrix_h = nearest_positive_definite(corr2cov(corr_matrix, np.full(fill_value=pred_harris_stdev / div_factor, shape=51)))
    cov_matrix_t = nearest_positive_definite(corr2cov(corr_matrix, np.full(fill_value=pred_trump_stdev / div_factor, shape=51)))
    harris_dist = scipy.stats.multivariate_normal.rvs(harris, cov=cov_matrix_h, size=10001)
    trump_dist = scipy.stats.multivariate_normal.rvs(trump, cov=cov_matrix_t, size=10001)
    # print('DEBUG:', harris_dist, trump_dist)
    margin_dist = harris_dist - trump_dist
    # print('DEBUG: margin_dist before concat', margin_dist)
    cd_margins = np.array(list(fundamentals_pred(['Nebraska CD-1', 'Nebraska CD-2', 'Nebraska CD-3', 'Maine CD-1', 'Maine CD-2'], return_samples=True).values()))
    # print('shapes: cd', cd_margins.shape, 'margin_dist', margin_dist.shape)
    # print('DEBUG: cd margins', cd_margins)
    margin_dist = np.append(np.transpose(margin_dist), cd_margins, axis=0)
    winner_dict = {}
    # print('DEBUG: margin_dist after concat', margin_dist)
    # print('DEBUG: margin_dist shape after concat', margin_dist.shape)
    for state, margins in zip(full_state_list, margin_dist):
        winner_dict[state] = margins
    # print(winner_dict)
    def harris_winner(margin):
        return 1 if margin > 0 else 0
    def trump_winner(margin):
        return 0 if margin > 0 else 1
    for state in full_state_list:
        state_ec = states_ec_dict[state]['ElectoralVotes']
        chance = np.mean(winner_dict[state] > 0)
        state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
        state_chances = pd.concat([state_chances, state_chance_df], axis=0)
        harris_winning = np.vectorize(harris_winner)(winner_dict[state])
        trump_winning = np.vectorize(trump_winner)(winner_dict[state])
        harris_ev = harris_winning * state_ec
        trump_ev = trump_winning * state_ec
        harris_ec += harris_ev
        trump_ec += trump_ev
#     for state in full_state_list:
#         state_data = cpvi[cpvi['State'] == state]
#         harris, trump = state_data['projected_harris_pct'].values[0], state_data['projected_trump_pct'].values[0]
#         harris_std, trump_std = pred_harris_stdev / div_factor, pred_trump_stdev / div_factor
# #         harris_dist = scipy.stats.t.rvs(df=5, loc=harris, scale=harris_std, 
# #                                         size=10001)
# #         trump_dist = scipy.stats.t.rvs(df=5, loc=trump, scale=trump_std, 
# #                                        size=10001)
        
#         chance = np.mean(harris_dist > trump_dist)
#         state_chance_df = pd.DataFrame({'chance': chance}, index=[state])
#         state_chances = pd.concat([state_chances, state_chance_df], axis=0)
#         def ec_adder(winning: bool):
#             return (states_ec_dict[state]['ElectoralVotes'] if winning else 0)
#         harris_ec += np.array([ec_adder(v) for v in (harris_dist > trump_dist)])
#         trump_ec += np.array([ec_adder(v) for v in (trump_dist > harris_dist)])
    chance = np.count_nonzero(harris_ec > trump_ec) / harris_ec.shape[0]
    pred = np.median(harris_ec)
    # print(harris_ec, trump_ec)
    return {'harris': np.mean(harris_ec > trump_ec), 'trump': np.mean(trump_ec > harris_ec), 'tie': np.mean(harris_ec == trump_ec)}, state_chances[1:], margin_dist

#############################################

polls_weight = 0.9
fund_weight = 0.1

polls_ev_pred = ev_pred()
fund_ev_pred, fund_preds, fund_samples = fundamentals_ev_pred()
polls_samples = np.array(list(samples.values()))
fund_margins = cpvi.set_index(['State'])['projected_margin']
fund_preds = pd.concat([fund_preds, fund_margins], axis=1).rename({'projected_margin':'margin'}, axis=1)

projected_margins = polls_weight * chances_df['margin'] + fund_weight * fund_preds['margin']
total_chance = polls_weight * chances_df['chance'] + fund_weight * fund_preds['chance']

# polls_ev_pred = ev_pred()
# fund_ev_pred, fund_preds = fundamentals_ev_pred()

harris_ev_win_chance = polls_weight * polls_ev_pred['harris'] + fund_weight * fund_ev_pred['harris']
trump_ev_win_chance = polls_weight * polls_ev_pred['trump'] + fund_weight * fund_ev_pred['trump']
tie_chance = polls_weight * polls_ev_pred['tie'] + fund_weight * fund_ev_pred['tie']

projection = pd.concat([projected_margins, total_chance], axis=1)
# projection

def winner(chance):
    # Dealing with Harris chances
    return 1 if chance > 0.5 else 0

proj_ev = projection.copy()
proj_ev['winner'] = proj_ev['chance'].map(winner)
proj_ev = proj_ev.merge(states_ec, left_on=proj_ev.index, right_on='state')
# harris_projected_evs = np.sum(proj_ev['winner'] * proj_ev['ElectoralVotes'])
# trump_projected_evs = 538 - harris_projected_evs

####################
####################

def simulate():
    def winner(margin):
        # 1 = Harris, 0 = Trump
        return 1 if margin > 0 else 0
    index = np.random.choice(10000, 1)[0]
    polls_scenario = polls_samples[:, index]
    # fund_scenario = fund_samples[:, index]
    fund_scenario = fund_preds['margin']
    scenario = polls_weight * polls_scenario + fund_weight * fund_scenario
    states_ec_dict_nonat = states_ec_dict.copy()
    states_ec_dict_nonat.pop('United States')
    harris_winning = np.sum(np.vectorize(winner)(scenario) * np.vectorize(lambda x: x['ElectoralVotes'])(np.array(list(states_ec_dict_nonat.values()))))
    scenario_df = pd.DataFrame({'state':full_state_list, 'margin':scenario})
    return scenario_df, harris_winning

def all_sims_ev():
    def winner(margin):
        # 1 = Harris, 0 = Trump
        return 1 if margin > 0 else 0
    harris_ev_sims = []
    states_ec_dict_nonat = states_ec_dict.copy()
    states_ec_dict_nonat.pop('United States')
    for index in range(polls_samples.shape[1]):
        polls_scenario = polls_samples[:, index]
        # fund_scenario = fund_samples[:, index]
        fund_scenario = fund_preds['margin']
        scenario = polls_weight * polls_scenario + fund_weight * fund_scenario
        harris_winning = np.sum(np.vectorize(winner)(scenario) * np.vectorize(lambda x: x['ElectoralVotes'])(np.array(list(states_ec_dict_nonat.values()))))
        harris_ev_sims.append(harris_winning)
    return harris_ev_sims

def find_tipping_point(one_sim: pd.Series):
    # Presuming that the index of series contains the states
    def winner(margin):
        # 1 = Harris, 0 = Trump
        return 1 if margin > 0 else 0
    states_ec_dict_nonat = states_ec_dict.copy()
    states_ec_dict_nonat.pop('United States')
    harris_winning = np.sum(np.vectorize(winner)(one_sim) * np.vectorize(lambda x: x['ElectoralVotes'])(np.array(list(states_ec_dict_nonat.values()))))
    if harris_winning > 269:
        winners_states = one_sim[one_sim > 0]
        winning_ec = harris_winning
    elif harris_winning < 269:
        winners_states = one_sim[one_sim < 0]
        winning_ec = 538 - harris_winning
    else:
        return 'N/A'
    winners_states = pd.DataFrame(winners_states.map(abs))
    winners_states = winners_states.sort_values(winners_states.columns.values[0], ascending=True)
    while winning_ec - states_ec_dict[winners_states.index.values[0]]['ElectoralVotes'] > 269:
        state = winners_states.index.values[0]
        winning_ec -= states_ec_dict[state]['ElectoralVotes']
        winners_states = winners_states[1:]
    state = winners_states.index.values[0]
    return state

# sim, _ = simulate()
# print(find_tipping_point(sim['margin']))

def tipping_point_frequencies(threshold=5):
    tipping_points = []
    def winner(margin):
        # 1 = Harris, 0 = Trump
        return 1 if margin > 0 else 0
    states_ec_dict_nonat = states_ec_dict.copy()
    states_ec_dict_nonat.pop('United States')
    for index in range(polls_samples.shape[1]):
        polls_scenario = polls_samples[:, index]
        # fund_scenario = fund_samples[:, index]
        fund_scenario = fund_preds['margin']
        scenario = polls_weight * polls_scenario + fund_weight * fund_scenario
        scenario_df = pd.DataFrame({'state':full_state_list, 'margin':scenario})
        tp_state = find_tipping_point(scenario_df['margin'])
        tipping_points.append(tp_state)
    tp_freq = (pd.Series(tipping_points).value_counts() / len(tipping_points) * 100)
    tp_freq = tp_freq[tp_freq > threshold]
    tp_freq_display = tp_freq.map(lambda x: f'{x:.2f}%')
    return tp_freq_display

tp_freq_display = tipping_point_frequencies()
tp_harris_chance = projection.loc[tp_freq_display.index.values[0], 'chance']


####################

def chance_rating(chance):
    # NOTE: chance MUST be Harris win probability!
    if chance >= 0.975:
        return 'Solid Harris'
    elif chance >= 0.75:
        return 'Likely Harris'
    elif chance >= 0.6:
        return 'Lean Harris'
    elif chance >= 0.4:
        return 'Tossup'
    elif chance >= 0.25:
        return 'Lean Trump'
    elif chance >= 0.025:
        return 'Likely Trump'
    else:
        return 'Solid Trump'

################### CHOROPLETH MAPS #######################################

###### POLLING PROJECTIONS #######

chances_for_choropleth = chances_df.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
chances_for_choropleth['trump_chance'] = 1 - chances_for_choropleth['chance']
chances_for_choropleth['Rating'] = chances_for_choropleth['chance'].map(chance_rating)
chances_for_choropleth['harris_chance_display'] = chances_for_choropleth['chance'].map(lambda x: f'{(x*100):.1f}%')
chances_for_choropleth['trump_chance_display'] = chances_for_choropleth['trump_chance'].map(lambda x: f'{(x*100):.1f}%')
fig_states_polling = px.choropleth(data_frame=chances_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                           color='chance',
                          color_continuous_scale='RdBu', range_color=[0, 1], hover_name='index', 
                          hover_data={'Abb_State':False, 'chance':False, 'harris_chance_display':True, 'trump_chance_display':True, 'Rating':True}, 
                          labels={'harris_chance_display':'Harris Win Chance', 'trump_chance_display':'Trump Win Chance'}, height=1000)
fig_states_polling.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projections - Polls-Only Model',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_states_polling.update_layout(coloraxis_colorbar=dict(
    title='Win Chance',
    tickvals=[0, 0.25, 0.5, 0.75, 1],
    ticktext=['Safe Trump', 'Likely Trump', 'Tossup', 'Likely Harris', 'Safe Harris']
))

fig_states_polling.update_traces(
    marker_line_color='black'
)

##

margins_for_choropleth = chances_df.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
margins_for_choropleth['margin_for_choropleth'] = margins_for_choropleth['margin'].map(lambda x: max(-15, min(x, 15)))
margins_for_choropleth['Rating'] = margins_for_choropleth['margin'].map(margin_rating)
margins_for_choropleth['Label'] = margins_for_choropleth['margin'].map(margin_with_party)
fig_states_polling_margins = px.choropleth(data_frame=margins_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                          color='margin_for_choropleth',
                          color_continuous_scale='RdBu', range_color=[-15, 15], hover_name='index', 
                          hover_data={'Abb_State':False, 'margin_for_choropleth':False, 'margin':False, 'Label':True, 'Rating':True},
                          labels={'Label':'Projected Margin'}, 
                          height=1000)
fig_states_polling_margins.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projected Margins - Polls-Only Model',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_states_polling_margins.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-15, -5, 0, 5, 15],
    ticktext=['>R+15', 'R+5', 'EVEN', 'D+5', '>D+15']
))

fig_states_polling_margins.update_traces(
    marker_line_color='black'
)

##

### FUNDAMENTALS ###

fund_chances_for_choropleth = fund_preds.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
fund_chances_for_choropleth['trump_chance'] = 1 - fund_chances_for_choropleth['chance']
fund_chances_for_choropleth['Rating'] = fund_chances_for_choropleth['chance'].map(chance_rating)
fund_chances_for_choropleth['harris_chance_display'] = fund_chances_for_choropleth['chance'].map(lambda x: f'{(x*100):.1f}%')
fund_chances_for_choropleth['trump_chance_display'] = fund_chances_for_choropleth['trump_chance'].map(lambda x: f'{(x*100):.1f}%')
fig_states_fund = px.choropleth(data_frame=fund_chances_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                           color='chance',
                          color_continuous_scale='RdBu', range_color=[0, 1], hover_name='index', 
                          hover_data={'Abb_State':False, 'chance':False, 'harris_chance_display':True, 'trump_chance_display':True, 'Rating':True}, 
                          labels={'harris_chance_display':'Harris Win Chance', 'trump_chance_display':'Trump Win Chance'}, height=1000)
fig_states_fund.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projections - Fundamentals-Only Model',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_states_fund.update_layout(coloraxis_colorbar=dict(
    title='Win Chance',
    tickvals=[0, 0.25, 0.5, 0.75, 1],
    ticktext=['Safe Trump', 'Likely Trump', 'Tossup', 'Likely Harris', 'Safe Harris']
))

fig_states_fund.update_traces(
    marker_line_color='black'
)

##

fund_margins_for_choropleth = fund_preds.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
fund_margins_for_choropleth['margin_for_choropleth'] = fund_margins_for_choropleth['margin'].map(lambda x: max(-15, min(x, 15)))
fund_margins_for_choropleth['Rating'] = fund_margins_for_choropleth['margin'].map(margin_rating)
fund_margins_for_choropleth['Label'] = fund_margins_for_choropleth['margin'].map(margin_with_party)
fig_states_fund_margins = px.choropleth(data_frame=fund_margins_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                           color='margin_for_choropleth',
                          color_continuous_scale='RdBu', range_color=[-15,  15], hover_name='index', 
                          hover_data={'Abb_State':False, 'margin_for_choropleth':False, 'margin':False, 'Label':True, 'Rating':True},
                          labels={'Label':'Projected Margin'}, 
                          height=1000)
fig_states_fund_margins.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projected Margins - Fundamentals-Only',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_states_fund_margins.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-15, -5, 0, 5, 15],
    ticktext=['>R+15', 'R+5', 'EVEN', 'D+5', '>D+15']
))

fig_states_fund_margins.update_traces(
    marker_line_color='black'
)

##### MODEL PROJECTION ######

projection_for_choropleth = projection.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
projection_for_choropleth['rounded_chance'] = projection_for_choropleth['chance'].map(lambda x: np.round(x, decimals=3))
projection_for_choropleth['trump_chance'] = 1 - projection_for_choropleth['rounded_chance']
projection_for_choropleth['harris_chance_display'] = projection_for_choropleth['rounded_chance'].map(lambda x: f'{(x*100):.1f}%')
projection_for_choropleth['trump_chance_display'] = projection_for_choropleth['trump_chance'].map(lambda x: f'{(x*100):.1f}%')
projection_for_choropleth['Rating'] = projection_for_choropleth['chance'].map(chance_rating)
fig_projection = px.choropleth(data_frame=projection_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                           color='chance',
                          color_continuous_scale='RdBu', range_color=[0, 1], hover_name='index', 
                          hover_data={'Abb_State':False, 'chance':False, 'harris_chance_display':True, 'trump_chance_display':True, 'Rating':True}, 
                          labels={'harris_chance_display':'Harris Win Chance', 'trump_chance_display':'Trump Win Chance'}, height=1000)
fig_projection.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projections - Fundamentals+Polls',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_projection.update_layout(coloraxis_colorbar=dict(
    title='Win Chance',
    tickvals=[0, 0.25, 0.5, 0.75, 1],
    ticktext=['Safe Trump', 'Likely Trump', 'Tossup', 'Likely Harris', 'Safe Harris']
))

fig_projection.update_traces(
    marker_line_color='black'
)

##

projected_margins_for_choropleth = projection.reset_index().merge(states_abb, left_on='index', right_on='Full_State').drop(['Full_State'], axis=1)
projected_margins_for_choropleth['margin_for_choropleth'] = projection_for_choropleth['margin'].map(lambda x: max(-15, min(x, 15)))
projected_margins_for_choropleth['Rating'] = projected_margins_for_choropleth['margin'].map(margin_rating)
projected_margins_for_choropleth['Label'] = projected_margins_for_choropleth['margin'].map(margin_with_party)
fig_projection_margins = px.choropleth(data_frame=projected_margins_for_choropleth, locations='Abb_State', locationmode='USA-states', 
                           color='margin_for_choropleth',
                          color_continuous_scale='RdBu', range_color=[-15, 15], hover_name='index', 
                          hover_data={'Abb_State':False, 'margin_for_choropleth':False, 'margin':False, 'Label':True, 'Rating':True},
                          labels={'Label':'Projected Margin'}, height=1000)
fig_projection_margins.update_layout(
    title_text = '2024 US Presidential Election SnoutCount Projected Margins - Fundamentals+Polls',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_projection_margins.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-15, -10, -5, 0, 5, 10, 15],
    ticktext=['>R+15', 'R+10', 'R+5', 'EVEN', 'D+5', 'D+10', '>D+15']
))

fig_projection_margins.update_traces(
    marker_line_color='black'
)

### PROJECTION HISTOGRAM ###

harris_ev_sims = all_sims_ev()
sims_df = pd.DataFrame(harris_ev_sims).rename({0:'ev'}, axis=1)
sims_df['winner'] = sims_df['ev'].map(lambda x: 'Harris win' if x > 269 else 'Trump win')
sims_df = sims_df.sort_values(['winner'], ascending=True)
fig_sims = px.histogram(data_frame=sims_df, x='ev', color='winner', labels={'ev':'EV Bin', 'count':'Sim Count'})
fig_sims.add_vline(x=270, line_dash='dot', annotation_text='Winning Threshold', annotation_position='top right')
fig_sims.update_layout(
    title_text='SnoutCount Combined Model Simulations (N=10,001)',
    xaxis_title='Electoral Votes',
    yaxis_title='Count',
    legend_title='Winner',
    template='plotly_dark'
)

########
harris_projected_evs = np.median(harris_ev_sims)
trump_projected_evs = 538 - harris_projected_evs