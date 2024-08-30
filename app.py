from dash import Dash, html, dcc, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import our data engineering and plot structuring file data_eng.py
from data_eng import *

app = Dash(__name__)

server = app.server

app.layout = html.Div(
    children=[
        html.H1(
            children='Centralized Aggregate and Model of Polls (CAMP)',
            style={'textAlign':'center', 'font-family':'Lucida Console'}
        ),
        html.H4(children='Last Updated: August 30, 2024', style={'textAlign':'center', 'font-family':'Lucida Console'}),
        html.Hr(),
        html.H2(children='Overview', style={'textAlign':'center', 'font-family':'Lucida Console'}),
        html.Div(
            children=[
                html.Div(children=[
                    html.H5(children='Polled Electoral College', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                    html.Div(children=f'Harris - {harris_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#100d94'}),
                    html.Div(children=f'Trump - {trump_polled_ev}', style={'textAlign':'center', 'font-family':'Lucida Console', 'color':'#940d0d'})
                ], className='box'),
                html.Div(children=[
                    html.H5(children='National Polling Average', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                    html.Div(children=nat_diff, style={'textAlign':'center', 'font-family':'Lucida Console', 'color':('#100d94' if avg_lowess_diff > 0 else '#940d0d')}),
                ], className='box'),
                html.Div(children=[
                    html.H5(children=f'Tipping Point Polling Average ({tp_state})', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                    html.Div(children=('Harris' if tp_margin >= 0 else 'Trump') + f'+{abs(tp_margin):.2f}%', style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                               'color':('#100d94' if tp_margin > 0 else '#940d0d')}),
                ], className='box'),
                html.Div(children=[
                    html.H5(children=f'Electoral College Bias', style={'textAlign':'center', 'font-family':'Lucida Console'}),
                    html.Div(children=ec_bias_pres, style={'textAlign':'center', 'font-family':'Lucida Console',
                                                                                                               'color':('#100d94' if ec_bias > 0 else '#940d0d')}),
                ], className='box')
            ]
        ),
        html.Hr(),
        html.H2(
            children='National Polling Averages and Trends, US Presidential 2024',
            style={'textAlign':'center', 'font-family':'Lucida Console'}
        ),
        dcc.Graph(
            id='polling-lowesses',
            figure=fig
        ),
        html.H4(children='National Polls Utilized', style={'textAlign':'center', 'font-family':'Lucida Console'}),
        dash_table.DataTable(
            data=nat_readable.sort_values(by=['Date'], ascending=False).to_dict('records'), page_size=10
        ),
        html.Hr(),
        html.H2(
            children='State Polling, US Presidential 2024',
            style={'textAlign':'center', 'font-family':'Lucida Console'}
        ),
        dcc.Graph(
            id='state-polling',
            figure=fig_states
        ),
        dcc.Graph(
            id='competitive-state-polling',
            figure=fig_comp
        ),
        html.Hr(),
        html.Div(
            children=['Polls dataset from ', dcc.Link(children=['538'], href='https://projects.fivethirtyeight.com/polls/president-general/2024/'), ' | See the code on ', dcc.Link(children=['Github'], href='https://github.com/Hackquantumcpp/camp')],
                      style={'textAlign':'center', 'font-family':'Lucida Console'}
        )
    ]
)

if __name__ == '__main__':
    app.run()