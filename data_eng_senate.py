import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import datetime
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
import json

warnings.filterwarnings('ignore')

generic = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/generic_ballot_polls.csv')

# So these are all national polls.
generic_np = generic[(generic['partisan'].isna()) & (generic['numeric_grade'] >= 2.0)]

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
    dup_polls_rv = duplicate_polls[duplicate_polls['population'] == 'rv']
    df_final = pd.concat([unique_polls, dup_polls_rv], axis=0)
    
    return df_final

generic_eng = pipeline(generic_np)
generic_used = generic_eng[['display_name', 'poll_id', 'end_date', 'population', 'sample_size', 'dem', 'rep', 'ind']].reset_index()
generic_used = generic_used.drop(['index'], axis=1)
generic_used['end_date'] = pd.to_datetime(generic_used['end_date'])
generic_used = generic_used[~generic_used['sample_size'].isna()]

generic_readable = generic_used.rename({'display_name':'Pollster', 'end_date':'Date', 'dem':'Democrats', 'rep':'Republican'},
                                      axis=1)
generic_readable['Sample'] = generic_readable['sample_size'].astype(int).astype(str) + ' ' + generic_readable['population'].map(lambda x: x.upper())
generic_readable = generic_readable.drop(['poll_id', 'population', 'sample_size', 'ind'], axis=1).set_index('Date').reset_index()


# LOWESS Curves - Generic
dem = generic_used['dem'].to_numpy()
rep = generic_used['rep'].to_numpy()
dates = generic_used['end_date'].to_numpy()
dem_lowess = lowess(dem, dates, frac=0.25, it=5, return_sorted=True)
rep_lowess = lowess(rep, dates, frac=0.25, it=5, return_sorted=True)

# Calculating margin of error (95% confidence intervals)
# Code from: https://james-brennan.github.io/posts/lowess_conf/
def smooth(x, y, xgrid):
    samples = np.random.choice(len(x), 50, replace=True)
    y_s = y[samples]
    x_s = x[samples]
    y_sm = lowess(y_s,x_s, frac=0.25, it=5,
                     return_sorted = False)
    # regularly sample it onto the grid
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, 
                                        fill_value='extrapolate')(xgrid)
    return y_grid

dates_range = pd.to_datetime(dem_lowess[:, 0])
dates_range_num = pd.to_numeric(dates_range)
dates_num = pd.to_numeric(dates)
K = 100
dem_smooths = np.stack([smooth(dates_num, dem, dates_range_num) for k in range(K)]).T
rep_smooths = np.stack([smooth(dates_num, rep, dates_range_num) for k in range(K)]).T
mean_d = np.nanmean(dem_smooths, axis=1)
stderr_d = scipy.stats.sem(dem_smooths, axis=1)
stderr_d = np.nanstd(dem_smooths, axis=1, ddof=0)
mean_r = np.nanmean(rep_smooths, axis=1)
stderr_r = scipy.stats.sem(rep_smooths, axis=1)
stderr_r = np.nanstd(rep_smooths, axis=1, ddof=0)
generic_curves = pd.DataFrame([dem_lowess[:, 1], rep_lowess[:, 1]]).T.rename({0:'Dem', 1:'Rep'}, axis=1).set_index(dates_range)

############## SENATE #########################

senate = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/senate_polls.csv')
senate['end_date'] = pd.to_datetime(senate['end_date'])
senate_recent = senate[senate['end_date'] >= pd.to_datetime('2024-05-01')]
# Only include people who are CURRENTLY running in the race
# Source: https://en.wikipedia.org/wiki/2024_United_States_Senate_elections
curr_running = ['Gallego', 'Lake', 'Heredia-Quintana', 'Garvey',
               'Schiff', 'Corey', 'Murphy', 'Paglino', 'Rochester',
               'Hansen', 'Katz', 'Bennett', 'Bonoan', 'Everidge',
               'Mucarsel-Powell', 'Nguyen', 'Scott', 'Billionaire',
               'Guiffre', 'Hirono', 'McDermott', 'Pohlman',
               'Banks', 'Horning', 'McCray', 'Costello', 'King',
               'Kouzounas', 'Alsobrooks', 'Hogan', 'Scott',
               'Deaton', 'Warren', 'Dern', 'Marsh', 'Rogers',
               'Slotkin', 'Solis-Mullen', 'Stein', 'Klobuchar',
               'White', 'Pinkins', 'Wicker', 'Hawley', 'Kunce',
               'Young', 'Daoud', 'Downey', 'Sheehy', 'Tester',
               'Eddy', 'Fischer', 'Osborn', 'Brown', 'Cunningham',
               'Destin', 'Hansen', 'Mazlo',' Rheinhart', 'Rosen',
               'Uehling', 'Bashaw', 'Kaplan', 'Khalil', 'Kim',
               'Kuniansky', 'Mooneyham', 'Domenici', 'Heinrich',
               'Gillibrand', 'Sapraicone', 'Christiansen',
               'Cramer', 'Brown', 'Moreno', 'Casey', 'McCormick',
               'McKay', 'Morgan', 'Whitehouse', 'Blackburn',
               'Chandler', 'Johnson', 'Moses', 'Robinson',
               'Allred', 'Cruz', 'Bowen', 'Curtis', 'Gleich',
               'Berry', 'Hill', 'Malloy', 'Sanders', 'Schoville',
               'Cao', 'Kaine', 'Cantwell', 'Garcia', 'Elliott',
               'Justice', 'Moran', 'Anderson', 'Baldwin',
               'Hovde', 'Leager', 'Barrasso', 'Morrow', 'Love',
               'Ricketts']
senate_np = senate_recent[(senate_recent['partisan'].isna()) & (senate_recent['numeric_grade'] >= 1.5)]
senate_np = senate_np[senate_np['answer'].isin(curr_running)]
party_with_cand = senate['answer'] + ' (' + senate['party'] + ')'
senate_np['party_with_cand'] = party_with_cand

def state_avgs_pipeline(senate_data: pd.DataFrame, state: str):
    state_race = senate_data[senate_data['state'] == state]# [senate['party'].isin(['DEM', 'REP'])]
    state_pivot = pd.pivot_table(data=state_race, values='pct', index=['poll_id', 'display_name', 'state', 'end_date', 'sample_size',
                                                               'population', 'numeric_grade'], columns=['party_with_cand'],
                                                              aggfunc='last', fill_value=0).reset_index()
    state_pivot = pipeline(state_pivot)
    
    # Sample size weights
    total_sample_size = np.sum(state_pivot['sample_size'])
    state_pivot['sample_size_weights'] = state_pivot['sample_size'].map(np.sqrt) / np.sqrt(np.median(state_pivot['sample_size']))
    state_pivot['sample_size_weights'] /= np.sum(state_pivot['sample_size_weights'])
    
    # Time weights
    # Variation of the equation used here: https://polls.votehub.us/
    today = datetime.datetime.today()
    delta = (today - state_pivot['end_date']).apply(lambda x: x.days)
    state_pivot['time_weights'] = (0.4 * (1 - delta/((today - state_pivot['end_date'].min()).days + 1)) + 
                                  0.6 *(0.3**(delta/((today - state_pivot['end_date'].min()).days + 1))))
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
        return 1
    state_pivot['population_weights'] = state_pivot['population'].map(population_weight)
    # state_pivot['population_weights'] /= np.sum(state_pivot['population_weights'])
    
    # Gather the weights together
    state_pivot['total_weights'] = state_pivot['sample_size_weights'] * state_pivot['time_weights'] * state_pivot['quality_weights'] * state_pivot['population_weights']
    state_pivot['total_weights'] /= np.sum(state_pivot['total_weights']) # Normalization step
    
    return state_pivot

senate_tp = senate_np[~senate_np['state'].isin(['Nebraska', 'Maine', 'Vermont'])]
state_list = senate_tp['state'].value_counts().index.to_numpy()
def contains_substring(str_list, substring):
    """
    Return all strings in a list of strings that contains a substring.
    """
    containing_strs = []
    for s in str_list:
        if substring in s:
            containing_strs.append(s)
    
    return containing_strs
def contains_substr_in_list(str_list, substr_list):
    """
    Returns all strings in a list of strings str_list that contains any substring in substr_list.
    """
    all_containing = []
    for substr in substr_list:
        containing = contains_substring(str_list, substr)
        all_containing.extend(containing)
    
    return all_containing

def all_polls_with_weights(senate_data):
    state_list = senate_data['state'].value_counts().index.to_numpy()
    polls_df = state_avgs_pipeline(senate_data, state_list[0])
    cols = contains_substr_in_list(polls_df.columns.values, ['DEM', 'REP'])
    polls_df = polls_df.rename({cols[0]:'DEM', cols[1]:'REP'}, axis=1)
    for state in state_list[1:]:
        pipelined_df = state_avgs_pipeline(senate_data, state)
        cols = contains_substr_in_list(pipelined_df.columns.values, ['DEM', 'REP'])
        pipelined_df = pipelined_df.rename({cols[0]:'DEM', cols[1]:'REP'}, axis=1)
        polls_df = pd.concat([polls_df, pipelined_df], axis=0)
    return polls_df

def all_polls_with_weights_ind(senate_data):
    state_list = senate_data['state'].value_counts().index.to_numpy()
    polls_df = state_avgs_pipeline(senate_data, state_list[0])
    cols = contains_substr_in_list(polls_df.columns.values, ['IND', 'DEM', 'REP'])
    cols = cols[1:]
    polls_df = polls_df.rename({cols[0]:'DEM', cols[1]:'REP'}, axis=1)
    for state in state_list[1:]:
        pipelined_df = state_avgs_pipeline(senate_data, state)
        cols = contains_substr_in_list(pipelined_df.columns.values, ['IND', 'DEM', 'REP'])
        if state == 'Nebraska':
            ne_regular = pipelined_df.drop(['Ricketts (REP)', 'Love (DEM)'], axis=1).rename({'Osborn (IND)':'DEM', 'Fischer (REP)':'REP'}, axis=1)
            ne_special = pipelined_df.drop(['Osborn (IND)', 'Fischer (REP)'], axis=1).rename({'Ricketts (REP)':'REP', 'Love (DEM)':'DEM'}, axis=1)
            ne_special.loc[0, 'state'] = 'Nebraska special'
            pipelined_df = pd.concat([ne_special, ne_regular], axis=0)
            polls_df = pd.concat([polls_df, pipelined_df], axis=0)
            continue
        if state == 'Maine':
            cols[1] = 'Kouzounas (REP)'
            pipelined_df = pipelined_df.drop(['Costello (DEM)'], axis=1)
        if state == 'Vermont':
            cols = cols[1:]
            pipelined_df = pipelined_df.drop(['Berry (IND)'], axis=1)
        pipelined_df = pipelined_df.rename({cols[0]:'DEM', cols[1]:'REP'}, axis=1)
        polls_df = pd.concat([polls_df, pipelined_df], axis=0)
    polls_df = polls_df.drop(['Hill (LIB)', 'Schoville (OTH)', 'Berry (IND)'], axis=1)
    return polls_df

senate_ind = senate_np[senate_np['state'].isin(['Nebraska', 'Maine', 'Vermont'])]
state_ind_polls = all_polls_with_weights_ind(senate_ind).drop(['time_weights', 'sample_size_weights', 'quality_weights', 'poll_id',
                                                              'numeric_grade'], axis=1)
state_polls = all_polls_with_weights(senate_tp)
state_polls = state_polls[['display_name', 'state', 'end_date', 'sample_size', 'population',
                          'DEM', 'REP', 'total_weights']]

senate_state_polls = pd.concat([state_polls, state_ind_polls], axis=0)

senate_state_polls['Sample'] = senate_state_polls['sample_size'].astype(int).astype(str) + ' ' + senate_state_polls['population'].apply(lambda x: x.upper())
senate_state_polls['total_weights'] = senate_state_polls['total_weights'].map(lambda x: f'{x:.5f}')
senate_state_polls = senate_state_polls.drop(['sample_size', 'population'], axis=1).rename(
    {'state':'State', 'display_name':'Pollster', 'total_weights':'Weights in State Polling Averages', 'end_date':'Date'}, axis=1
).reset_index().drop(['index'], axis=1)[['Date', 'Pollster', 'State', 'Sample', 'DEM', 'REP', 
                                        'Weights in State Polling Averages']]


def get_state_averages(senate_data, state):
    dem_avgs = []
    rep_avgs = []
    pipelined_df = state_avgs_pipeline(senate_data, state).replace({'NA':0})
    cols = contains_substr_in_list(pipelined_df.columns.values, ['DEM', 'REP'])
    dem_avg = np.sum(pipelined_df[cols[0]] * pipelined_df['total_weights'])
    rep_avg = np.sum(pipelined_df[cols[1]] * pipelined_df['total_weights'])
    
    return [state, dem_avg, rep_avg, cols[0], cols[1]]

state_avgs = []

for state in state_list:
    avg = get_state_averages(senate_tp, state)
    state_avgs.append(avg)

ne_df = state_avgs_pipeline(senate_np, 'Nebraska')
ne_avgs = ['Nebraska', np.sum(ne_df['total_weights'] * ne_df['Osborn (IND)']), np.sum(ne_df['total_weights'] * ne_df['Fischer (REP)']),
          'Osborn (IND)', 'Fischer (REP)']
ne_avgs_spec = ['Nebraska special', np.sum(ne_df['total_weights'] * ne_df['Love (DEM)']), np.sum(ne_df['total_weights'] * ne_df['Ricketts (REP)']),
               'Love (DEM)', 'Ricketts (REP)']
me_df = state_avgs_pipeline(senate_np, 'Maine')
me_avgs = ['Maine', np.sum(me_df['total_weights'] * me_df['King (IND)']), np.sum(me_df['total_weights'] * me_df['Kouzounas (REP)']),
          'King (IND)', 'Kouzounas (REP)']
vt_df = state_avgs_pipeline(senate_np, 'Vermont')
vt_avgs = ['Vermont', np.sum(vt_df['total_weights'] * vt_df['Sanders (IND)']), np.sum(vt_df['total_weights'] * vt_df['Malloy (REP)']),
          'Sanders (IND)', 'Malloy (REP)']
state_avgs.extend([ne_avgs, ne_avgs_spec, me_avgs, vt_avgs])
state_averages_df = pd.DataFrame(state_avgs).rename({0:'state', 1:'DEM', 2:'REP', 3:'DEM_cand', 4:'REP_cand'}, axis=1)
dem_cands = state_averages_df['DEM_cand'].str.extract(r'(.+)\([A-za-z]{3}\)')
rep_cands = state_averages_df['REP_cand'].str.extract(r'(.+)\([A-za-z]{3}\)')
state_averages_df['DEM_cand'] = dem_cands
state_averages_df['REP_cand'] = rep_cands

with open('data/other/us-states-senate-2024.json', 'r') as file:
    senate_map = json.load(file)

senate_json_df = pd.read_json('data/other/us-states-senate-2024.json')
id_with_states = senate_json_df['features'].astype(str).str.extract(r"\{'type': 'Feature', 'id': '(\d{2})', 'properties': {'name': '([A-za-z\s]*)', .*\}")
id_with_states = id_with_states.rename({0:'id', 1:'state'}, axis=1)

state_averages_df = state_averages_df.merge(id_with_states, on='state')
# By convention, positive margins indicate Dem advantage, while negative margins indicate Rep advantage.
state_averages_df['Margin'] = state_averages_df['DEM'] - state_averages_df['REP']

from data_eng_pres import margin_rating, margin_with_party

state_averages_df['Rating'] = state_averages_df['Margin'].map(margin_rating)
state_averages_df['Label'] = state_averages_df['Margin'].map(margin_with_party)
state_averages_df_all = state_averages_df.copy()
state_averages_df['margin_for_choropleth'] = state_averages_df['Margin'].map(lambda x: min(x, 20))
state_averages_df = state_averages_df[state_averages_df['state'] != 'Nebraska special']
states_abb = pd.read_csv('data/other/Electoral_College.csv').drop(['Electoral_College_Votes'], axis=1)
state_averages_df = state_averages_df.merge(states_abb, left_on='state', right_on='Full_State')

# For overview cards
generic_margin = dem_lowess[:, 1][-1] - rep_lowess[:, 1][-1]
generic_margin_label = ('D' if generic_margin > 0 else 'R') + f'+{generic_margin:.2f}%'

winner = state_averages_df_all['Margin'].map(lambda x: 'DEM' if x > 0 else 'REP')
state_averages_df_all['Winner'] = winner
dem_polled_sen_seats = 33 + np.count_nonzero(winner == 'DEM')
rep_polled_sen_seats = 41 + np.count_nonzero(winner == 'REP')
# By convention, positive = DEM win and negative = REP win
sen_margin = dem_polled_sen_seats - rep_polled_sen_seats
total = 100
sen_margin /= (total / 100)

from data_eng_pres import harris_polled_ev, trump_polled_ev

pres_winner = 'DEM' if harris_polled_ev > trump_polled_ev else 'REP'

def senate_tipping_point():
    if sen_margin > 0:
        sen_winner = 'DEM'
        curr = dem_polled_sen_seats
    elif sen_margin < 0:
        sen_winner = 'REP'
        curr = rep_polled_sen_seats
    else: # sen_margin = 0
        sen_winner = pres_winner
        curr = dem_polled_sen_seats if pres_winner == 'DEM' else rep_polled_sen_seats
    winners_states = state_averages_df_all[state_averages_df_all['Winner'] == sen_winner].sort_values(['Margin'], ignore_index=True, ascending=True)
    threshold = 50 if sen_winner == pres_winner else 51
    while curr - 1 > threshold:
        winners_states = winners_states[1:]
        curr -= 1
    return winners_states.loc[0, 'state']

sen_tp_state = senate_tipping_point()
sen_tp_margin = state_averages_df_all[state_averages_df_all['state'] == sen_tp_state]['Margin'].values[0]
senate_bias = sen_tp_margin - generic_margin

# Plots/charts

fig_line = px.line(data_frame=generic_curves, x=generic_curves.index, y=['Dem', 'Rep'], title='Generic Congressional Ballot Polling',
                  markers=False)

fig_dem_CI = go.Figure([
    go.Scatter(
        name='Dem CI Upper Bound',
        x = dates_range,
        y = mean_d + 1.96*stderr_d,
        mode='lines',
        marker=dict(color='#8972fc'),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    go.Scatter(
        name='Dem CI Lower Bound',
        x = dates_range,
        y = mean_d - 1.96*stderr_d,
        mode='lines',
        marker=dict(color='#8972fc'),
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(137, 114, 252, 0.15)',
        showlegend=False,
        hoverinfo='skip'
    )
    
])

fig_rep_CI = go.Figure([
    go.Scatter(
        name='Rep CI Upper Bound',
        x = dates_range,
        y = mean_r + 1.96*stderr_r,
        mode='lines',
        marker=dict(color='#fc7472'),
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    go.Scatter(
        name='Rep CI Lower Bound',
        x = dates_range,
        y = mean_r - 1.96*stderr_r,
        mode='lines',
        marker=dict(color='#fc7472'),
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(252, 116, 114, 0.15)',
        showlegend=False,
        hoverinfo='skip'
    )
    
])

fig_scatter = px.scatter(data_frame=generic_used, x='end_date', y=['dem', 'rep'], opacity=0.3)
                        # trendline='lowess', trendline_options=dict(frac=0.2))

fig = go.Figure(data=fig_line.data + fig_scatter.data + fig_dem_CI.data + fig_rep_CI.data)

# fig.update_traces(hovertemplate=None)

fig.update_layout(
    title='Generic Congressional Ballot Polling',
    xaxis_title = 'Date',
    yaxis_title = 'Polled Vote %',
    legend_title = 'Legend',
)

fig_senate = px.choropleth(
    data_frame=state_averages_df,
    locations='Abb_State',
    locationmode='USA-states',
    color='margin_for_choropleth',
    color_continuous_scale='RdBu',
    range_color=[-20, 20],
    hover_name='state', 
    hover_data={'Abb_State':False, 'Rating':True, 'Margin':False, 'Label':True, 
                'margin_for_choropleth':False},
    labels={'Label':'Average Margin'},
    width=1400,
    height=1000
)

fig_senate.update_layout(
    title_text = '2024 US Senate Election State Polling Averages',
    geo_scope='usa', # limit map scope to USA
)

fig_senate.update_layout(coloraxis_colorbar=dict(
    title='Margin',
    tickvals=[-20, -10, 0, 10, 20],
    ticktext=['>R+20', 'R+10', 'EVEN', 'D+10', '>D+20']
))