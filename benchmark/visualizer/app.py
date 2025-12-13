import os
import time
from dash import Dash
import dash_bootstrap_components as dbc
from benchmark.visualizer.pages.general_components import create_navigation
import dash
from dash import html, dcc

def create_app():
    """Create and configure the Dash application"""
    
    app = Dash(__name__, 
              external_stylesheets=[
                  dbc.themes.BOOTSTRAP,
                  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css'
              ],
              use_pages=True,
              pages_folder='pages') 
    
    app.title = "MLKV+ Benchmark"
    
    # Set up layout with navigation
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='navigation-container'),
        html.Div(dash.page_container, style={'marginTop': '80px'})
    ])
    
    @app.callback(
        dash.Output('navigation-container', 'children'),
        dash.Input('url', 'pathname')
    )
    def update_navigation(pathname):
        return create_navigation(pathname)
    
    return app