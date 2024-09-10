from dash import Dash, html, dcc, dash_table, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
# import datetime

# Import our data engineering and plot structuring files
import data_eng_pres as de
import data_eng_senate as sen

app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

server = app.server

##### OVERVIEW INFOCARDS #####
polled_ev_card =dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='Polled Electoral College', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Harris - {de.harris_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#100d94'},
                                id='harris-ev'),
            html.Div(children=f'Trump - {de.trump_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#940d0d'},
                                id='trump-ev')
        ]
    ), style={'width':'18rem'}, color=('primary' if de.harris_polled_ev > de.trump_polled_ev else 'danger'), outline=True
)

nat_avg_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children='National Polling Average', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=de.nat_diff, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#100d94' if de.avg_lowess_diff > 0 else '#940d0d')},
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
                                                                                                                'color':('#100d94' if de.tp_margin > 0 else '#940d0d')},
                                                                                                                id='tp_avg'),
        ], style={'width':'18rem'}
    ), color=('primary' if de.tp_margin > 0 else 'danger'), outline=True
)

ec_bias_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Electoral College Bias', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=de.ec_bias_pres, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#100d94' if de.ec_bias > 0 else '#940d0d')},
                                                                                                                id='ec-bias')
        ],  style={'width':'18rem'}
    ), color=('primary' if de.ec_bias > 0 else 'danger'), outline=True
)

generic_ballot_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Generic Congressional Popular Polling Average', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=sen.generic_margin_label, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#100d94' if sen.generic_margin > 0 else '#940d0d')},
                                                                                                                id='generic-margin')
        ], style={'width':'18rem'}
    ), color=('primary' if sen.generic_margin > 0 else 'danger'), outline=True
)

polled_senate_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Polled Senate Seats', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(children=f'Democrats - {sen.dem_polled_sen_seats}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#100d94'},
                   id='dem-senate-seats'),
            html.Div(children=f'Republicans - {sen.rep_polled_sen_seats}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#940d0d'},
                     id='rep-senate-seats')
        ], style={'width':'18rem'}
    ), color=('primary' if ((sen.dem_polled_sen_seats > sen.rep_polled_sen_seats) or (sen.dem_polled_sen_seats == sen.rep_polled_sen_seats
                                                                                     and de.harris_polled_ev > de.trump_polled_ev)) else 'danger'), outline=True
)

senate_bias_card = dbc.Card(
    dbc.CardBody(
        [
            html.H6(children=f'Senate Bias', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.P(children=('D' if sen.senate_bias > 0 else 'R') + f'+{abs(sen.senate_bias):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                            'color':('#100d94' if sen.senate_bias > 0 else '#940d0d')},
                   id='senate-bias')
        ], style={'width':'18rem'}
    ), color=('primary' if sen.senate_bias > 0 else 'danger'), outline=True
)

##############################


def serve_layout():
    return html.Div(
        children=[
            html.Br(),
            html.H1(
                children='Centralized Aggregate and Model of Polls (CAMP)',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            # dcc.Interval(
            #     id='interval-component',
            #     interval=1*1000, # every second, for debug purposes
            #     n_intervals=0
            # ),
            html.H4(children=f'Last updated: September 10, 2024 12:02 AM UTC', style={'textAlign':'center', 'font-family':'Lucida Console'}, id='last-updated'),
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
                            dbc.Col(senate_bias_card, width='auto')
                        ], style={'justify-content':'center'}
                    )
                ]
            ),
            html.Hr(),
            html.H2(
                children='National Polling Averages and Trends, US Presidential 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='polling-lowesses',
                figure=de.fig
            ),
            html.H4(children='National Polls Utilized', style={'textAlign':'center', 'font-family':'Lucida Console'}),
            html.Div(dbc.Table.from_dataframe(
                de.nat_readable.sort_values(by=['Date'], ascending=False), striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            ), style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.H2(
                children='State Polling, US Presidential 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='state-polling',
                figure=de.fig_states
            ),
            dcc.Graph(
                id='competitive-state-polling',
                figure=de.fig_comp
            ),
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
            html.H2(
                children='National Polling, US Congressional 2024 - Generic',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dcc.Graph(
                id='generic-ballot-polling',
                figure=sen.fig
            ),
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
            html.H2(
                children='State Polling, US Senate 2024',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            dbc.Alert(
                [
                    html.P("Note: The prominent independent candidates in Maine, Vermont and Nebraska are counted as Democratic due to them either being incumbent senators caucusing with the Democratic Party (Maine, Vermont), or having the support of state Democrats (Nebraska). The Nebraska special election is not shown in the map below. If you're really anxious or curious about that, it's a Solid R.")
                ],
                color='info',
                style={'font-family':'Lucida Console', 'justify-content':'center'}
            ),
            dcc.Graph(
                id='senate-polling',
                figure=sen.fig_senate
            ),
            html.H4(
                children='State Polls Utilized',
                style={'textAlign':'center', 'font-family':'Lucida Console'}
            ),
            html.Div(dbc.Table.from_dataframe(
                sen.senate_state_polls.sort_values(by=['Date'], ascending=False), striped=True, bordered=True, hover=True, 
                responsive=True,
                style={'font-family':'monospace'}, 
            ), style={'maxHeight':'400px', 'overflow':'scroll'}),
            html.Hr(),
            html.Div(
                children=['Polls dataset from ', dcc.Link(children=['538'], href='https://projects.fivethirtyeight.com/polls/president-general/2024/'), ' | See the code on ', dcc.Link(children=['Github'], href='https://github.com/Hackquantumcpp/camp')],
                        style={'textAlign':'center', 'font-family':'Lucida Console'}
            )
        ]
    )

app.layout = serve_layout

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
#     return {'textAlign':'center', 'font-family':'Lucida Console', 'color':('#100d94' if de.avg_lowess_diff > 0 else '#940d0d')}

# @callback(
#     Output('nat-avg', 'children'),
#     Input('interval-component', 'n_intervals')
# )
# def update_tp_state(n):
#     return f'Tipping Point Polling Average ({de.tp_state})'

if __name__ == '__main__':
    app.run()