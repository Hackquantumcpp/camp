import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy
import datetime
import re

governor = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/governor_polls.csv')

party_with_cand = governor['answer'] + ' (' + governor['party'] + ')'
governor['party_with_cand'] = party_with_cand
governor['end_date'] = pd.to_datetime(governor['end_date'])
governor_recent = governor[governor['end_date'] >= pd.to_datetime('2024-04-01')]

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

for state in governor_np['state'].value_counts().index.values:
    avg = get_state_averages(governor_np, state)
    state_avgs.append(avg)

state_avgs_df = pd.DataFrame(state_avgs)
state_avgs_df = state_avgs_df.rename({0:'state', 1:'DEM', 2:'REP', 3:'DEM_cand', 4:'REP_cand'}, axis=1)

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
                'margin_for_choropleth':False},
    labels={'Label':'Average Margin'},
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

