from dash import Dash, html, dcc, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our data engineering and plot structuring file data_eng.py
from data_eng import *

app = Dash(__name__)

server = app.server

app.layout = html.Div(
    children=[
        html.H1(
            children='Centralized Aggregate and Mean of Polls (CAMP)',
            style={'textAlign':'center', 'font-family':'Lucida Console'}
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
        
        html.H2(
            children='State Polling, US Presidential 2024',
            style={'textAlign':'center', 'font-family':'Lucida Console'}
        ),
        dcc.Graph(
            id='state-polling',
            figure=fig_states
        ),
        html.Hr(),
        html.Div(
            children=['Polls dataset from ', dcc.Link(children=['538'], href='https://projects.fivethirtyeight.com/polls/president-general/2024/'), ' | See the code on ', dcc.Link(children=['Github'], href='https://github.com/Hackquantumcpp/camp')],
                      style={'textAlign':'center', 'font-family':'Lucida Console'}
        )
    ]
)

if __name__ == '__main__':
    app.run(debug=True)