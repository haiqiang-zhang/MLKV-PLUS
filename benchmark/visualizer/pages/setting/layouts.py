from dash import register_page
from visualizer.BenchmarkRunner import BenchmarkRunner
import dash_bootstrap_components as dbc
from dash import html, dcc
from visualizer.pages.setting.components import create_setting_card




def create_setting_layout():
    """Create the benchmark page layout"""
    return dbc.Container([
        html.H1("Setting", className="my-4"),
        create_setting_card(),
    ])



    
# Set the layout
layout = create_setting_layout() 
    
# Register this page
register_page(__name__, path='/setting', title="Setting", layout=layout)
