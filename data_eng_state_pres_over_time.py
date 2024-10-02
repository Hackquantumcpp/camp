import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import datetime
import warnings
from data_eng_pres import pipeline, polls_for_state_avgs, competitive

def state_avgs_pipeline(state: str, date: datetime.date):
    state_race = polls_for_state_avgs[polls_for_state_avgs['state'] == state]# [senate['party'].isin(['DEM', 'REP'])]
    state_pivot = pd.pivot_table(data=state_race, values='pct', index=['poll_id', 'display_name', 'state', 'end_date', 'sample_size',
                                                               'population', 'numeric_grade'], columns=['candidate_name'],
                                                              aggfunc='last', fill_value='NA').reset_index()
    state_pivot['end_date'] = pd.to_datetime(state_pivot['end_date'])
    state_pivot = state_pivot[(state_pivot['end_date'] >= pd.to_datetime('2024-07-24')) & (state_pivot['end_date'] <= date)]
    state_pivot = pipeline(state_pivot)
    
    total_num_polls = state_pivot.shape[0]

    # Sample size weights
    total_sample_size = np.sum(state_pivot['sample_size'])
    state_pivot['sample_size_weights'] = (state_pivot['sample_size'].map(np.sqrt) / np.sqrt(np.median(state_pivot['sample_size'])))
    state_pivot['sample_size_weights'] /= np.sum(state_pivot['sample_size_weights'])
    
    # Time weights
    # Variation of the equation used here: https://polls.votehub.us/
    latest_date = date
    delta = (latest_date - state_pivot['end_date']).apply(lambda x: x.days)
    linear_weights = (1 - delta/((latest_date - state_pivot['end_date'].min()).days + 1))
    exp_weights = 0.4**(delta/((latest_date - state_pivot['end_date'].min()).days + 1))
    state_pivot['time_weights'] =  0.3 * linear_weights + 0.7 * exp_weights
    # state_pivot['time_weights'] /= np.sum(state_pivot['time_weights'])
    
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
    # state_pivot['quality_weights'] /= np.sum(state_pivot['quality_weights'])
    
    # Population weights
    def population_weight(population):
        if population == 'a':
            return 0.6
        elif population == 'rv':
            return 0.9
        return 1
    state_pivot['population_weights'] = state_pivot['population'].map(population_weight)
    # state_pivot['population_weights'] /= np.sum(state_pivot['population_weights'])

    # Gather the weights together
    state_pivot['total_weights'] = state_pivot['sample_size_weights'] * state_pivot['time_weights'] * state_pivot['quality_weights'] * state_pivot['population_weights']
    state_pivot['total_weights'] /= np.sum(state_pivot['total_weights']) # Normalization step
    
    return state_pivot

def get_state_average_over_time(state):
    dem_avgs = []
    rep_avgs = []

    state_race = polls_for_state_avgs[polls_for_state_avgs['state'] == state]# [senate['party'].isin(['DEM', 'REP'])]
    state_pivot = pd.pivot_table(data=state_race, values='pct', index=['poll_id', 'display_name', 'state', 'end_date', 'sample_size',
                                                               'population', 'numeric_grade'], columns=['candidate_name'],
                                                              aggfunc='last', fill_value='NA').reset_index()
    state_pivot['end_date'] = pd.to_datetime(state_pivot['end_date'])
    state_pivot = state_pivot[(state_pivot['end_date'] >= pd.to_datetime('2024-07-24'))]
    state_pivot = pipeline(state_pivot)

    date_range = pd.date_range(start=state_pivot['end_date'].min(), end=datetime.datetime.today(), freq='d', inclusive='both')
    for date in date_range:
        pipelined_df = state_avgs_pipeline(state, date).replace({'NA':0})
        dem_avg = np.sum(pipelined_df['Kamala Harris'] * pipelined_df['total_weights'])
        rep_avg = np.sum(pipelined_df['Donald Trump'] * pipelined_df['total_weights'])
        dem_avgs.append(dem_avg)
        rep_avgs.append(rep_avg)
    
    return pd.DataFrame({'Date':date_range,'Kamala Harris':dem_avgs, 'Donald Trump':rep_avgs})

def get_state_timeseries(state_list):
    state_series = {}
    for state in state_list:
        state_ts = get_state_average_over_time(state)
        state_series.update({state: state_ts})
    return state_series

state_timeseries = get_state_timeseries(competitive['state'].values)