from dash import Dash, html, dcc, dash_table, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
# import datetime

# Import our data engineering and plot structuring files
import data_eng_pres as de
import data_eng_senate as sen
import data_eng_gub as gub
from data_eng_state_pres_over_time import state_timeseries
# import model as mod

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc_css])

server = app.server

##### OVERVIEW INFOCARDS #####
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

###### OVERVIEW - SWING STATE CARDS ######

def display_margin(state):
    # print(de.states[de.states['state'] == state])
    disp_margin = abs(de.states[de.states['state'] == state]['Average Polling Margin'].values[0])
    return ('Harris' if de.states[de.states['state'] == state]['Kamala Harris'].values[0] > de.states[de.states['state'] == state]['Donald Trump'].values[0] 
                             else 'Trump') + f'+{disp_margin:.2f}%'

def winner_color(state):
    return ('primary' if de.states[de.states['state'] == state]['Kamala Harris'].values[0] > de.states[de.states['state'] == state]['Donald Trump'].values[0] 
                             else 'danger')

north_carolina_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5(children='North Carolina', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            dbc.CardImg(src=app.get_asset_url('north-carolina-outline.png'), style={'height':'15%', 'width':'auto'}),
            html.H6(children=display_margin('North Carolina'),
                             style={'textAlign':'center', 'font-family':'Lucida Console'})
        ], style={'width':'18rem', 'height':'auto', 'textAlign':'center'}
    ), color = winner_color('North Carolina'), outline=False
)

georgia_card = dbc.Card(
    dbc.CardBody(
        [
            html.H5(children='Georgia', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            dbc.CardImg(src=app.get_asset_url('georgia-outline.png'), style={'height':'15%', 'width':'auto'}),
            html.H6(children=display_margin('Georgia'),
                             style={'textAlign':'center', 'font-family':'Lucida Console'})
        ], style={'width':'18rem', 'height':'auto', 'textAlign':'center'}
    ), color = winner_color('Georgia'), outline=False
)


##############################


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
            html.H4(children=f'Last updated: October 2, 2024 7:20 PM UTC', style={'textAlign':'center', 'font-family':'Lucida Console'}, id='last-updated'),
            # html.H4(children=f'Debug: {str(datetime.datetime.now())}', style={'textAlign':'center', 'font-family':'Lucida Console'}, id='debug-last-updated'),
            html.Hr(),
            html.H2(children='Overview', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Br(),
            html.Div(
                children=[
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
                    hoverData={'points': [{'y': 'Pennsylvania'}]},
                    style={'width':'50%', 'display':'inline-block'}
                ),
                dcc.Graph(
                    id='state-timeseries',
                    style={'width':'50%', 'display':'inline-block'}
                )
            ]),
            html.Br(),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Dropdown(
                options=['All', 'Pennsylvania', 'Georgia', 'Arizona', 'North Carolina', 'Michigan', 'Wisconsin', 'Nevada', 'Maine CD-2', 'Texas',
                         'Florida', 'Ohio', 'Minnesota', 'New Hampshire', 'Nebraska CD-2'],
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
            html.Br(),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Div(dbc.Table.from_dataframe(
                gub.state_polls, striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            ), style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
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
    Output(component_id='state-timeseries', component_property='figure'),
    Input(component_id='competitive-state-polling', component_property='hoverData')
)
def state_timeseries_fetch(hoverData):
    state = hoverData['points'][0]['y'] # ['y']# [0]['customdata']
    # print(state)
    timeseries_df = state_timeseries[state]
    fig_line = px.line(data_frame=timeseries_df, x='Date', y=['Kamala Harris', 'Donald Trump'], title=state)
    # fig_scatter = px.scatter(data_frame=de.state_readable[de.state_readable['Date'] >= pd.to_datetime('2024-07-24')][de.state_readable['State'] == state].set_index('Date'), y=['Kamala Harris', 'Donald Trump'], opacity=0.5)
    fig = go.Figure(data=fig_line.data)# + fig_scatter.data)
    fig.update_layout(
        title=f'{state} Polling Average',
        xaxis_title='Date',
        yaxis_title='Polled Vote %',
        template='plotly_dark'
    )
    return fig

# Live updates

# def update():
#     importlib.reload(de)

# @callback(
#     Output('last-updated', 'children'),
#     Input('interval-component', 'n_intervals')
# )

# def last_updated(n):
#     update()
#     return f'Last updated: {str(datetime.datetime.now())}'

# @callback(
#     Output('debug-last-updated', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def debug_lu(n):
#     return f'Debug: {str(datetime.datetime.now())}'

# @callback(
#     Output('harris-ev', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def update_harris_ev(n):
#     return f'Harris - {de.harris_polled_ev}'

# @callback(
#     Output('trump-ev', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def update_trump_ev(n):
#     return f'Trump - {de.trump_polled_ev}'

# @callback(
#     Output('nat-avg', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def update_nat_avg(n):
#     return de.nat_diff

# @callback(
#     Output('nat-avg', 'style'),
#     Input('interval-component', 'n_intervals')
# )
# def update_nat_avg_style(n):
#     return {'textAlign':'center', 'font-family':'Lucida Console', 'color':('#05c9fa' if de.avg_lowess_diff > 0 else '#ff4a3d')}

# @callback(
#     Output('nat-avg', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def update_tp_state(n):
#     return f'Tipping Point Polling Average ({de.tp_state})'

if __name__ == '__main__':
    app.run()