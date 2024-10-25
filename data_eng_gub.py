import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy
import datetime
import warnings
import json

warnings.filterwarnings('ignore')

banned_pollsters = pd.read_csv('data/other/banned_pollsters.csv')['banned_pollsters']

governor = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/governor_polls.csv')

party_with_cand = governor['answer'] + ' (' + governor['party'] + ')'
governor['party_with_cand'] = party_with_cand
governor['end_date'] = pd.to_datetime(governor['end_date'])
governor_recent = governor[governor['end_date'] >= pd.to_datetime('2024-04-01')]
governor_recent = governor_recent[~governor_recent['pollster_rating_name'].isin(banned_pollsters.values)]

# Only those currently running
# From: https://en.wikipedia.org/wiki/2024_United_States_gubernatorial_elections#Race_summary
curr_running = ['Meyer', 'Ramone', 'Braun', 'McCormick', 'Kehoe',
               'Quade', 'Slantz', 'Rainwater', 'Busse', 'Gianforte',
               'Leib', 'Ayotte', 'Craig', 'Robinson', 'Ross',
               'Stein', 'Turner', 'Armstrong', 'Coachman', 'Piepkorn',
               'Cox', 'King', 'Latham', 'Tomeny', 'Williams',
               'Blais', 'Charlestin', 'Mutino', 'Scott',
               'Ferguson', 'Reichert', 'Kolenich', 'Linko-Looper',
               'Morrisey', 'Williams', 'Dalmau', 'González-Colón',
               'Ortiz', 'Jiménez']

governor_np = governor_recent[(governor_recent['partisan'].isna()) & (governor_recent['numeric_grade'] >= 1.5)]
governor_np = governor_np[governor_np['answer'].isin(curr_running)]

from data_eng_senate import pipeline, state_avgs_pipeline, contains_substring, contains_substr_in_list, all_polls_with_weights, get_state_averages

state_polls = all_polls_with_weights(governor_np)[['display_name', 'state', 'end_date', 'sample_size', 'population',
                          'DEM', 'REP', 'total_weights']]
state_polls['Sample'] = state_polls['sample_size'].astype(int).astype(str) + ' ' + state_polls['population'].map(lambda x: x.upper())
state_polls = state_polls.rename({'display_name':'Pollster', 'state':'State', 'end_date':'Date',
                                 'total_weights':'Weight for State Polling Average'}, axis=1).drop(
    ['sample_size', 'population'], axis=1)[['Date', 'Pollster', 'State', 'Sample', 'DEM', 'REP', 'Weight for State Polling Average']]

state_polls = state_polls.sort_values(['Date'], ascending=False)
def round_to_five(x):
    return f'{x:.5f}'
state_polls['Weight for State Polling Average'] = state_polls['Weight for State Polling Average'].map(round_to_five)

state_avgs = []
state_list = governor_np['state'].value_counts().index.values
state_list = np.delete(state_list, np.where(state_list == 'Puerto Rico'))

for state in state_list:
    avg = get_state_averages(governor_np, state)
    state_avgs.append(avg)

state_avgs_df = pd.DataFrame(state_avgs)
state_avgs_df = state_avgs_df.rename({0:'state', 1:'DEM', 2:'REP', 3:'DEM_cand', 4:'REP_cand'}, axis=1)

dem_cands = state_avgs_df['DEM_cand'].str.extract(r'(.+)\([A-za-z]{3}\)')
rep_cands = state_avgs_df['REP_cand'].str.extract(r'(.+)\([A-za-z]{3}\)')
state_avgs_df['DEM_cand'] = dem_cands
state_avgs_df['REP_cand'] = rep_cands

# By convention, for margin, positive -> Dem advantage, negative -> Rep advantage
state_avgs_df['Margin'] = state_avgs_df['DEM'] - state_avgs_df['REP']

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
    return direc + '+' + f'{abs(margin):.1f}%'

state_avgs_df['Rating'] = state_avgs_df['Margin'].map(margin_rating)
state_avgs_df['Label'] = state_avgs_df['Margin'].map(margin_with_party)
state_avgs_df['margin_for_choropleth'] = state_avgs_df['Margin'].map(lambda x: min(x, 20))
states_abb = pd.read_csv('data/other/Electoral_College.csv').drop(['Electoral_College_Votes'], axis=1)
state_avgs_df = state_avgs_df.merge(states_abb, left_on='state', right_on='Full_State')

# Puerto Rico

def pr_avg_pipeline(senate_data: pd.DataFrame, state: str):
    state_race = senate_data[senate_data['state'] == state]# [senate['party'].isin(['DEM', 'REP'])]
    state_pivot = pd.pivot_table(data=state_race, values='pct', index=['poll_id', 'display_name', 'state', 'end_date',
                                                               'population', 'numeric_grade'], columns=['party_with_cand'],
                                                              aggfunc='last', fill_value=0).reset_index()
    state_pivot = pipeline(state_pivot)

    ## Looser weights until/unless more polls come
    
    # Sample size weights
    # total_sample_size = np.sum(state_pivot['sample_size'])
    # state_pivot['sample_size_weights'] = state_pivot['sample_size'].map(lambda x: np.sqrt(min(x, 5000))) / np.sqrt(np.median(state_pivot['sample_size'].map(lambda x: min(x, 5000))))
    # state_pivot['sample_size_weights'] /= np.sum(state_pivot['sample_size_weights'])
    
    # Time weights
    # Variation of the equation used here: https://polls.votehub.us/
    latest_date = datetime.datetime.today()
    delta = (latest_date - state_pivot['end_date']).apply(lambda x: x.days)
    timerange = (latest_date - state_pivot['end_date'].min()).days
    state_pivot['time_weights'] = (0.4 * (1 - delta/(timerange + 1)) + 
                                  0.6 *(0.3**(delta/(timerange + 1))))
    # state_pivot['time_weights'] /= np.sum(state_pivot['time_weights'])
    
    # Quality weights
    min_quality = 1.9
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
    state_pivot['total_weights'] = state_pivot['time_weights'] * state_pivot['quality_weights'] * state_pivot['population_weights'] # * state_pivot['sample_size_weights']
    state_pivot['total_weights'] /= np.sum(state_pivot['total_weights']) # Normalization step
    
    return state_pivot

puerto_rico_df = pr_avg_pipeline(governor_np, 'Puerto Rico')
dalmau = np.sum(puerto_rico_df['total_weights'] * puerto_rico_df['Dalmau (PRI)'])
gonzalez = np.sum(puerto_rico_df['total_weights'] * puerto_rico_df['González-Colón (NPP)'])
jimenez = np.sum(puerto_rico_df['total_weights'] * puerto_rico_df['Jiménez (OTH)'])
ortiz = np.sum(puerto_rico_df['total_weights'] * puerto_rico_df['Ortiz (PPD)'])
puerto_rico_avgs = pd.DataFrame({'candidate': ['Dalmau (PIP)', 'González-Colón (PNP)', 'Ortiz (PPD)', 'Jiménez (PD)'], 'pct': [dalmau, gonzalez, ortiz, jimenez], 
                                 'color':['green', 'blue', 'red', 'cyan']})
puerto_rico_avgs = puerto_rico_avgs.sort_values(by='pct', ascending=True)

# Plots

fig_governor = px.choropleth(
    data_frame=state_avgs_df,
    locations='Abb_State',
    locationmode='USA-states',
    color='margin_for_choropleth',
    color_continuous_scale='RdBu',
    range_color=[-20, 20],
    hover_name='state', 
    hover_data={'Abb_State':False, 'Rating':True, 'Margin':False, 'Label':True, 
                'margin_for_choropleth':False, 'DEM_cand':True, 'REP_cand':True},
    labels={'Label':'Average Margin', 'DEM_cand':'Democrat', 'REP_cand':'Republican'},
    height=1000
)

fig_governor.update_layout(
    title_text = '2024 US Gubernatorial Election State Polling Averages',
    geo_scope='usa', # limit map scope to USA
    template='plotly_dark'
)

fig_governor.update_geos(
    showland=True, landcolor="#777778"
)

fig_governor.update_traces(
    marker_line_color='black'
)

fig_governor.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-20, -10, 0, 10, 20],
    ticktext=['>R+20', 'R+10', 'EVEN', 'D+10', '>D+20']
))

fig_pr = px.bar(data_frame=puerto_rico_avgs, x='pct', y='candidate', color='color', hover_data={'color':False}, labels={'pct':'Vote %', 'candidate':'Candidate'})

fig_pr.update_layout(
    title='Puerto Rico Gubernatorial Election',
    yaxis_title='Candidate',
    template='plotly_dark',
    showlegend=False
)


puerto_rico_map_df = puerto_rico_avgs.iloc[0, :].copy()
puerto_rico_map_df['state'] = 'Puerto Rico'
puerto_rico_map_df = pd.DataFrame(puerto_rico_map_df).T

with open('data/other/puertorico.geojson', 'r') as file:
    pr_map = json.load(file)

fig_pr_map = px.choropleth(
    data_frame=puerto_rico_map_df,
    locations='state',
    featureidkey='properties.id',
    geojson=pr_map,
    color='color',
    hover_name='state',
    hover_data={'color':False},
    labels={'candidate':'Winner'},
)

fig_pr_map.update_geos(fitbounds='locations', visible=False)

fig_pr_map.update_layout(template='plotly_dark')


