from dash import register_page
from visualizer.BenchmarkRunner import BenchmarkRunner
import dash_bootstrap_components as dbc
from dash import html, dcc
from visualizer.pages.results.components import (
    create_results_card,
    create_visualization_config_card,
    create_chart_card
)
from visualizer.pages.general_components import create_results_loading_card

from visualizer.callbacks.visualization_callbacks import register_visualization_callbacks
from visualizer.callbacks.results_callbacks import register_results_callbacks

def create_results_layout(result_options):
    """Create the results and visualization page layout"""
    return dbc.Container([
        html.H1("Results & Visualization", className="my-4"),
        
        # Results Loading
        create_results_loading_card(result_options, prefix="results"),
        
        # Results Table
        create_results_card(),
        
        # Visualization Configuration
        create_visualization_config_card(),
        
        # Chart
        create_chart_card(),
        
        # Store for selected results
        dcc.Store(id='selected-results', data=[])
    ]) 



# Create visualizer instance and get available results
runner = BenchmarkRunner()


# Set the layout
layout = create_results_layout(runner.get_available_results()) 

# Register this page
register_page(__name__, path='/results', title="Results & Visualization", layout=layout)

register_visualization_callbacks(runner)
register_results_callbacks(runner)