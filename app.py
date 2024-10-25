from dash import Dash, html, dcc, dash_table, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
# import datetime

# Import our data engineering, plot structuring, and model files
import data_eng_pres as de
import data_eng_senate as sen
import data_eng_gub as gub
from data_eng_state_pres_over_time import state_timeseries
from data_eng_senate_seat_over_time import senate_timeseries
import snoutcount_model as scm

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc_css])
app.title = 'Centralized Aggregate and Model of Polls'

server = app.server

##### OVERVIEW INFOCARDS #####
election_chances_card =dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='Election Win Chances', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Harris - {scm.harris_ev_win_chance * 100:.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'},
                                id='harris-chance'),
            html.Div(children=f'Trump - {scm.trump_ev_win_chance * 100:.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'},
                                id='trump-chance'),
            html.Div(children=f'Tie - {scm.tie_chance * 100:.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'white'},
                                id='tie-chance')
        ]
    ), style={'width':'18rem'}, color=('primary' if scm.harris_ev_win_chance > scm.trump_ev_win_chance else 'danger'), outline=True
)

projected_ev_card =dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='Median Projected Electoral College', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Harris - {scm.harris_projected_evs}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'},
                                id='harris-ev-projection'),
            html.Div(children=f'Trump - {scm.trump_projected_evs}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'},
                                id='trump-ev-projection')
        ]
    ), style={'width':'18rem'}, color=('primary' if scm.harris_projected_evs > scm.trump_projected_evs else 'danger'), outline=True
)

tp_chances_card =dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Win Chances in Most Frequent Tipping Point ({scm.tp_freq_display.index.values[0]})', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Harris - {scm.tp_harris_chance * 100:.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'},
                                id='harris-tp-chance'),
            html.Div(children=f'Trump - {100 - (scm.tp_harris_chance * 100):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'},
                                id='trump-tp-chance')
        ]
    ), style={'width':'18rem'}, color=('primary' if scm.tp_harris_chance > 0.5 else 'danger'), outline=True
)

polled_ev_card =dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='Polled Electoral College', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Harris - {de.harris_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'},
                                id='harris-ev'),
            html.Div(children=f'Trump - {de.trump_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'},
                                id='trump-ev')
        ]
    ), style={'width':'18rem'}, color=('primary' if de.harris_polled_ev > de.trump_polled_ev else 'danger'), outline=True
)

nat_avg_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='National Polling Average', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=de.nat_diff, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if de.avg_lowess_diff > 0 else '#ff4a3d')},
                                id='nat-avg'),
        ], style={'width':'18rem'}
    ), color=('primary' if de.avg_lowess_diff > 0 else 'danger'), outline=True
)

tp_avg_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Tipping Point Polling Average ({de.tp_state})', style={'textAlign':'center', 'font-family':'Lucida Console'},
                                id='tipping_point'),
            html.P(children=('Harris' if de.tp_margin >= 0 else 'Trump') + f'+{abs(de.tp_margin):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                                'color':('#05c9fa' if de.tp_margin > 0 else '#ff4a3d')},
                                                                                                                id='tp_avg'),
        ], style={'width':'18rem'}
    ), color=('primary' if de.tp_margin > 0 else 'danger'), outline=True
)

ec_bias_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Electoral College Bias', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=de.ec_bias_pres, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if de.ec_bias > 0 else '#ff4a3d')},
                                                                                                                id='ec-bias')
        ],  style={'width':'18rem'}
    ), color=('primary' if de.ec_bias > 0 else 'danger'), outline=True
)

generic_ballot_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Generic Congressional Popular Polling Average', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=sen.generic_margin_label, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if sen.generic_margin > 0 else '#ff4a3d')},
                                                                                                                id='generic-margin')
        ], style={'width':'18rem'}
    ), color=('primary' if sen.generic_margin > 0 else 'danger'), outline=True
)

polled_senate_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Polled Senate Seats', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Democrats - {sen.dem_polled_sen_seats}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'},
                   id='dem-senate-seats'),
            html.Div(children=f'Republicans - {sen.rep_polled_sen_seats}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'},
                     id='rep-senate-seats')
        ], style={'width':'18rem'}
    ), color=('primary' if ((sen.dem_polled_sen_seats > sen.rep_polled_sen_seats) or (sen.dem_polled_sen_seats == sen.rep_polled_sen_seats
                                                                                     and de.harris_polled_ev > de.trump_polled_ev)) else 'danger'), outline=True
)

senate_tp_avg_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Senate Tipping Point Polling Average ({sen.sen_tp_state})', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=(sen.state_averages_df_all[sen.state_averages_df_all['state'] == sen.sen_tp_state][('DEM_cand' if sen.sen_tp_margin >= 0 else 'REP_cand')].values[0].strip()) + f'+{abs(sen.sen_tp_margin):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                                'color':('#05c9fa' if sen.sen_tp_margin > 0 else '#ff4a3d')},
                                                                                                                id='senate_tp_avg'
                   )
        ], style={'width':'18rem'}
    ), color=('primary' if sen.sen_tp_margin >= 0 else 'danger'), outline=True
)

senate_bias_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Senate Bias', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=('D' if sen.senate_bias > 0 else 'R') + f'+{abs(sen.senate_bias):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                            'color':('#05c9fa' if sen.senate_bias > 0 else '#ff4a3d')},
                   id='senate-bias')
        ], style={'width':'18rem'}
    ), color=('primary' if sen.senate_bias > 0 else 'danger'), outline=True
)

app.layout = html.Div(
        children=[
            html.Br(),
            html.H2(
                children='Centralized Aggregate and Model of Polls (CAMP)',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            # dcc.Interval(
            #     id='interval-component',
            #     interval=1*1000, # every second, for debug purposes
            #     n_intervals=0
            # ),
            html.H4(children=f'Last updated: October 25, 2024 8:40 PM UTC', style={'textAlign':'center', 'font-family':'Lucida Console'}, id='last-updated'),
            # html.H4(children=f'Debug: {str(datetime.datetime.now())}', style={'textAlign':'center', 'font-family':'Lucida Console'}, id='debug-last-updated'),
            html.Hr(),
            html.H2(children='Overview', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Br(),
            html.Div(
                children=[
                    dbc.Row(
                        [
                            dbc.Col(election_chances_card, width='auto'),
                            dbc.Col(projected_ev_card, width='auto'),
                            dbc.Col(tp_chances_card, width='auto')
                        ], style={'justify-content':'center'}
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(polled_ev_card, width='auto'),
                            dbc.Col(nat_avg_card, width='auto'),
                            dbc.Col(tp_avg_card, width='auto'),
                            dbc.Col(ec_bias_card, width='auto')
                        ], style={'justify-content':'center'}
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col(generic_ballot_card, width='auto'),
                            dbc.Col(polled_senate_card, width='auto'),
                            dbc.Col(senate_tp_avg_card, width='auto'),
                            dbc.Col(senate_bias_card, width='auto')
                        ], style={'justify-content':'center'}
                    )
                ]
            ),
            # html.Div(
            #     children=[
            #         dbc.Row(
            #             [
            #                 dbc.Col(north_carolina_card, width='auto'), # TEST
            #                 dbc.Col(georgia_card, width='auto')
            #             ], style={'justify-content':'center'}
            #         )
            #     ]
            # ),
            html.Hr(),
            html.H3(children='SnoutCount Election Prediction Model', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Br(),
            dbc.Tabs(
                [
                    dbc.Tab([
                    html.Br(), html.H4(children=f"{'Harris' if scm.harris_ev_win_chance > scm.trump_ev_win_chance else 'Trump'} is leading with a {max(scm.harris_ev_win_chance, scm.trump_ev_win_chance) * 100:.1f}% chance of winning the election in the SnoutCount combined fundamentals+polls model.",
                        style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if scm.harris_ev_win_chance > scm.trump_ev_win_chance else '#ff4a3d')}),
                    html.Br(),
                    dcc.Graph(
                        id='projection',
                        figure=scm.fig_projection,
                        style={'justify':'center', 'width':'auto'}
                    ),
                    dcc.Graph(
                        id='sims-histogram',
                        figure=scm.fig_sims,
                        style={'justify':'center', 'width':'auto'}
                    ),
                    html.Div(dbc.Table.from_dataframe(
                        pd.DataFrame(scm.tp_freq_display).reset_index().rename({'count':'Tipping Point Probability', 'index':'Most Frequent Tipping Point States'}, axis=1), striped=True, bordered=True, hover=True, responsive=True,
                        style={'font-family':'monospace', 'width':'49%', 'margin':'auto'}
                    )),], label='Combined'),
                    dbc.Tab([
                        html.Br(),
                        html.H4(children=f"{'Harris' if scm.polls_ev_pred['harris'] > scm.polls_ev_pred['trump'] else 'Trump'} is leading with a {max(scm.polls_ev_pred['harris'], scm.polls_ev_pred['trump']) * 100:.1f}% chance of winning the election in the SnoutCount polls-only model.",
                            style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if scm.polls_ev_pred['harris'] > scm.polls_ev_pred['trump'] else '#ff4a3d')}),
                        html.Br(),
                        dcc.Graph(
                            id='polls-only',
                            figure=scm.fig_states_polling,
                            style={'justify':'center', 'width':'auto'}
                        )
                    ], label='Polls-Only'),
                    dbc.Tab([html.Br(),
                        html.H4(children=f"{'Harris' if scm.fund_ev_pred['harris'] > scm.fund_ev_pred['trump'] else 'Trump'} is leading with a {max(scm.fund_ev_pred['harris'], scm.fund_ev_pred['trump']) * 100:.1f}% chance of winning the election in the SnoutCount fundamentals-only model.",
                            style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if scm.fund_ev_pred['harris'] > scm.fund_ev_pred['trump'] else '#ff4a3d')}),
                        # html.H5(children='*Utilizing uncorrelated sampling, thus may not match map below.', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                        html.Br(),
                        dcc.Graph(
                            id='fundamentals-only',
                            figure=scm.fig_states_fund,
                            style={'justify':'center', 'width':'auto'}
                        )], label='Fundamentals-Only'
                    ),
                    dbc.Tab([html.Br(),
                        html.H4(children=f"Press the button to simulate the election with the SnoutCount combined fundamentals+polls model.",
                            style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'white'}),
                        html.H5(children="Note that not every outcome will necessarily be likely! In fact, many of them won't be. Outliers are inevitable when running statistical simulations.",
                                style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'white'}),
                        html.Div(dbc.Button("Simulate!", id='simulate-button', n_clicks=0, outline=True, color='primary'), className='d-grid gap-2 col-6 mx-auto'), # style={'width':'18rem', 'height':'auto', 'textAlign':'center'}
                        # html.H5(children='*Utilizing uncorrelated sampling, thus may not match map below.', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                        html.Br(),
                        html.Div(id='harris_sim_ev', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'}),
                        html.Div(id='trump_sim_ev', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'}),
                        html.Div(id='sim_polling_error', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'}),
                        html.Div(id='sim_tipping_point', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                        html.Br(),
                        dcc.Graph(
                            id='simulation',
                            style={'justify':'center', 'width':'auto'}
                        )], label='Simulate The Election'
                    ),
                    
                ]
            ),
            html.Hr(),
            html.H3(
                children='National Polling Averages and Trends, US Presidential 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='polling-lowesses',
                figure=de.fig,
                className='dbc'
            ),
            html.Br(),
            html.H4(children='National Polls Utilized', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(dbc.Table.from_dataframe(
                de.nat_readable.sort_values(by=['Date'], ascending=False), striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            ), style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.H3(
                children='State Polling, US Presidential 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='state-polling',
                figure=de.fig_states,
                style={'justify':'center', 'width':'auto'},
            ),
            html.Div([
                dcc.Graph(
                    id='competitive-state-polling',
                    figure=de.fig_comp,
                    hoverData={'points': [{'y': de.tp_state}]},
                    style={'width':'50%', 'display':'inline-block'}
                ),
                dcc.Graph(
                    id='state-timeseries',
                    style={'width':'50%', 'display':'inline-block'}
                )
            ]),
            html.Br(),
            # dcc.RadioItems(
            #     options=['Candidates', 'Margin'], value='Candidates', id='state-timeseries-radio-items'
            # ),
            # html.Br(),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Dropdown(
                options=['All', 'Pennsylvania', 'Georgia', 'Arizona', 'North Carolina', 'Michigan', 'Wisconsin', 'Nevada', 'Maine CD-2', 'Texas',
                         'Florida', 'Ohio', 'Minnesota', 'New Hampshire', 'Nebraska CD-2', 'Alaska'],
                value='All',
                id='state-filter',
                # inline=True,
                searchable=True,
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            # html.Br(),
            html.Div(id='state-polls-table', style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.H3(
                children='National Polling, US Congressional 2024 - Generic',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='generic-ballot-polling',
                figure=sen.fig,
            ),
            html.Br(),
            html.H4(
                children='Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Div(dbc.Table.from_dataframe(
                sen.generic_readable.sort_values(by=['Date'], ascending=False), striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            ), style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.H3(
                children='State Polling, US Senate 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dbc.Alert(
                [
                    html.P("Note: The prominent independent candidates in Maine, Vermont and Nebraska are counted as Democratic due to them either being incumbent senators caucusing with the Democratic Party (Maine, Vermont), or having the support of state Democrats (Nebraska). The Nebraska special election is not shown in the map below. If you're really anxious or curious about that, it's a Solid R.")
                ],
                color='secondary',
                style={'font-family':'Lucida Console', 'justify-content':'center'}
            ),
            dcc.Graph(
                id='senate-polling',
                figure=sen.fig_senate,
                style={'justify':'center', 'width':'auto'},
            ),
            html.Div([
                dcc.Graph(
                    id='competitive-senate-polling',
                    figure=sen.fig_comp,
                    hoverData={'points': [{'y': 'Ohio'}]},
                    style={'width':'50%', 'display':'inline-block'}
                ),
                dcc.Graph(
                    id='senate-timeseries',
                    style={'width':'50%', 'display':'inline-block'}
                )
            ]),
            html.Br(),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Dropdown(
                options=['All', 'Montana', 'Ohio', 'Nebraska', 'Florida', 'Texas', 'Michigan', 'Pennsylvania', 'Wisconsin', 'Arizona', 'Nevada', 'Maryland'],
                value='All',
                id='senate-state-filter',
                # inline=True,
                searchable=True,
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Div(id='senate-state-polls-table', style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.H3(
                children='State Polling, US Gubernatorial 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='gubernatorial-polling',
                figure = gub.fig_governor,
                style={'justify':'center', 'width':'auto'}
            ),
            # html.Div([
            #     dcc.Graph(
            #         id='puerto-rico-governor',
            #         figure=gub.fig_pr,
            #         style={'display':'inline-block', 'width':'50%'}
            #     ),
            #     # dcc.Graph(
            #     #     id='puerto-rico-map',
            #     #     figure=gub.fig_pr_map,
            #     #     style={'display':'inline-block', 'width':'50%'}
            #     # ),
            # ]),
            dcc.Graph(
                id='puerto-rico-governor',
                figure=gub.fig_pr,
                style={'justify':'center', 'width':'auto'}
            ),
            html.Br(),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Dropdown(
                options=['All', 'North Carolina', 'New Hampshire', 'Washington', 'Missouri'],
                value='All',
                id='governor-state-filter',
                # inline=True,
                searchable=True,
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Div(style={'maxHeight':'400px', 'overflow':'scroll'}, id='governor-table'),
            html.Hr(),
            html.Br(),
            html.Div(
                children=['Polls dataset from ', dcc.Link(children=['538'], href='https://projects.fivethirtyeight.com/polls/president-general/2024/'), ' | See the code on ', dcc.Link(children=['Github'], href='https://github.com/Hackquantumcpp/camp')],
                        style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Br()
        ], className='dbc'
    )


@callback(
    Output(component_id='state-polls-table', component_property='children'),
    Input(component_id='state-filter', component_property='value')
)
def filter_state_polls_table(val):
    if val == 'All':
        data = de.state_readable[de.state_readable['Date'] >= pd.to_datetime('2024-07-24')].sort_values(by=['Date'], ascending=False)# .to_dict('records')
    else:
        data = de.state_readable[de.state_readable['Date'] >= pd.to_datetime('2024-07-24')][de.state_readable['State'] == val].sort_values(by=['Date'], ascending=False)# .to_dict('records')
    return dbc.Table.from_dataframe(
                df=data, striped=True, bordered=True, hover=True, responsive=True, style={'font-family':'monospace'}
            )

@callback(
    Output(component_id='senate-state-polls-table', component_property='children'),
    Input(component_id='senate-state-filter', component_property='value')
)
def filter_senate_polls_table(val):
    if val == 'All':
        data = sen.senate_state_polls.sort_values(by=['Date'], ascending=False)
    else:
        data = sen.senate_state_polls[sen.senate_state_polls['State'] == val].sort_values(by=['Date'], ascending=False)
    return dbc.Table.from_dataframe(
                df=data, striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            )

@callback(
    Output(component_id='governor-table', component_property='children'),
    Input(component_id='governor-state-filter', component_property='value')
)
def filter_governor_polls_table(val):
    if val == 'All':
        data = gub.state_polls.sort_values(by=['Date'], ascending=False)
    else:
        data = gub.state_polls[gub.state_polls['State'] == val].sort_values(by=['Date'], ascending=False)
    return dbc.Table.from_dataframe(
                df=data, striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            )

@callback(
    Output(component_id='state-timeseries', component_property='figure'),
    Input(component_id='competitive-state-polling', component_property='hoverData'),
    # Input(component_id='state-timeseries-radio-items', component_property='value')
)
def state_timeseries_fetch(hoverData):
    state = hoverData['points'][0]['y'] # ['y']# [0]['customdata']
    # print(state)
    timeseries_df = state_timeseries[state]
    fig = px.line(data_frame=timeseries_df, x='Date', y=['Kamala Harris', 'Donald Trump'], title=state)
    # fig_scatter = px.scatter(data_frame=de.state_readable[de.state_readable['Date'] >= pd.to_datetime('2024-07-24')][de.state_readable['State'] == state].set_index('Date'), y=['Kamala Harris', 'Donald Trump'], opacity=0.5)
    # fig = go.Figure(data=fig_line.data)# + fig_scatter.data)
    fig.update_traces(hovertemplate=None)
    fig.update_layout(
        title=f'{state} Polling Average',
        xaxis_title='Date',
        yaxis_title='Polled Vote %',
        template='plotly_dark',
        hovermode='x unified'
    )
    return fig

@callback(
    Output(component_id='senate-timeseries', component_property='figure'),
    Input(component_id='competitive-senate-polling', component_property='hoverData'),
    # Input(component_id='state-timeseries-radio-items', component_property='value')
)
def senate_timeseries_fetch(hoverData):
    state = hoverData['points'][0]['y'] # ['y']# [0]['customdata']
    # print(state)
    timeseries_df = senate_timeseries[state]
    fig = px.line(data_frame=timeseries_df, x='Date', y=['DEM', 'REP'], title=state)
    # fig_scatter = px.scatter(data_frame=de.state_readable[de.state_readable['Date'] >= pd.to_datetime('2024-07-24')][de.state_readable['State'] == state].set_index('Date'), y=['Kamala Harris', 'Donald Trump'], opacity=0.5)
    # fig = go.Figure(data=fig_line.data)# + fig_scatter.data)
    fig.update_traces(hovertemplate=None)
    fig.update_layout(
        title=f'{state} Senate Seat Polling Average',
        xaxis_title='Date',
        yaxis_title='Polled Vote %',
        template='plotly_dark',
        hovermode='x unified'
    )
    return fig

@callback(
    Output(component_id='simulation', component_property='figure'),
    Output(component_id='harris_sim_ev', component_property='children'),
    Output(component_id='trump_sim_ev', component_property='children'),
    Output(component_id='sim_polling_error', component_property='children'),
    Output(component_id='sim_tipping_point', component_property='children'),
    Input(component_id='simulate-button', component_property='n_clicks')
)
def simulate_election(n_clicks):
    scenario_df, harris_sim_ev = scm.simulate()
    scenarios_df_choro = scenario_df.reset_index().merge(scm.states_abb, left_on='state', right_on='Full_State').drop(['Full_State'], axis=1)
    scenarios_df_choro['margin_for_choropleth'] = scenarios_df_choro['margin'].map(lambda x: max(-15, min(x, 15)))
    scenarios_df_choro['Rating'] = scenarios_df_choro['margin'].map(scm.margin_rating)
    scenarios_df_choro['Label'] = scenarios_df_choro['margin'].map(scm.margin_with_party)
    polling_errors = (scm.chances_df['margin'] - scenarios_df_choro.set_index('state')['margin'])
    sim_polling_error = np.mean(polling_errors)
    sim_polling_error_display = ('Harris' if sim_polling_error > 0 else 'Trump') + f' Overestimated by {abs(sim_polling_error):.2f}%'
    scenarios_df_choro = scenarios_df_choro.set_index(['state'])
    scenarios_df_choro['polling_errors'] = polling_errors
    scenarios_df_choro = scenarios_df_choro.reset_index()
    scenarios_df_choro['polling_error_display'] = scenarios_df_choro['polling_errors'].map(lambda x: ('Harris' if x > 0 else 'Trump') + f' Overestimated by {abs(x):.2f}%')
    scenarios_df_choro['winner'] = scenarios_df_choro['margin'].map(lambda x: 'Harris' if x > 0 else 'Trump')
    scenarios_df_choro = scenarios_df_choro.sort_values(['winner'], ascending=True)
    sim_tipping_point = scm.find_tipping_point(scenario_df.set_index(['state'])['margin'])
    
    fig_scenario_margins = px.choropleth(data_frame=scenarios_df_choro, locations='Abb_State', locationmode='USA-states', 
                            color='margin_for_choropleth',
                            color_continuous_scale='RdBu', range_color=[-15, 15], hover_name='state', 
                            hover_data={'Abb_State':False, 'margin_for_choropleth':False, 'margin':False, 'Label':True, 'Rating':True, 'polling_error_display':True},
                            labels={'Label':'Projected Margin', 'polling_error_display':'State Polling Error'}, height=1000)
    fig_scenario_margins.update_layout(
        title_text = 'Simulated 2024 US Presidential Election',
        geo_scope='usa', # limit map scope to USA
        template='plotly_dark'
    )

    fig_scenario_margins.update_layout(coloraxis_colorbar=dict(
        title='Margin',
        tickvals=[-15, -10, -5, 0, 5, 10, 15],
        ticktext=['>R+15', 'R+10', 'R+5', 'EVEN', 'D+5', 'D+10', '>D+15']
    ))

    fig_scenario_margins.update_traces(
        marker_line_color='black'
    )

    fig_scenario_no_margins = px.choropleth(data_frame=scenarios_df_choro, locations='Abb_State', locationmode='USA-states', 
                            color='winner',
                            color_continuous_scale='RdBu', range_color=[-15, 15], hover_name='index', 
                            hover_data={'Abb_State':False, 'margin_for_choropleth':False, 'margin':False, 'Label':True, 'Rating':True, 'polling_error_display':True},
                            labels={'winner':'Winner', 'Label':'Projected Margin', 'polling_error_display':'State Polling Error'}, height=1000)
    fig_scenario_no_margins.update_layout(
        title_text = 'Simulated 2024 US Presidential Election',
        geo_scope='usa', # limit map scope to USA
        template='plotly_dark'
    )

    fig_scenario_no_margins.update_layout(coloraxis_colorbar=dict(
        title='Margin',
        tickvals=[-15, -10, -5, 0, 5, 10, 15],
        ticktext=['>R+15', 'R+10', 'R+5', 'EVEN', 'D+5', 'D+10', '>D+15']
    ))

    fig_scenario_no_margins.update_traces(
        marker_line_color='black'
    )

    harris_ev_stat = html.H5(children=f'Harris - {harris_sim_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#05c9fa'})
    trump_ev_stat = html.H5(children=f'Trump - {538 - harris_sim_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#ff4a3d'})
    sim_polling_error_stat = html.H5(children='Average Polling Error - ' + sim_polling_error_display, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if sim_polling_error > 0 else '#ff4a3d')})
    sim_tipping_point_stat = html.H5(children='Tipping Point State - ' + sim_tipping_point, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if harris_sim_ev > 269 else '#ff4a3d')})

    return fig_scenario_margins, harris_ev_stat, trump_ev_stat, sim_polling_error_stat, sim_tipping_point_stat

if __name__ == '__main__':
    app.run()