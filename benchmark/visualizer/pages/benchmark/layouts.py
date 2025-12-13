from dash import register_page
from visualizer.BenchmarkRunner import BenchmarkRunner
import dash_bootstrap_components as dbc
from dash import html, dcc
from visualizer.pages.benchmark.components import (
    create_status_card,
    create_benchmark_config_card
)
from visualizer.pages.general_components import create_results_loading_card, create_new_results_modal

from visualizer.callbacks.benchmark_callbacks import register_benchmark_callbacks
from visualizer.callbacks.status_callbacks import register_status_callbacks




def create_benchmark_layout(result_options):
    """Create the benchmark page layout"""
    return dbc.Container([
        html.H1("Benchmark Configuration", className="my-4"),
        
        # Results Loading
        create_results_loading_card(result_options, prefix="benchmark"),
        create_new_results_modal(prefix="benchmark"),
        
        
        # Benchmark Configuration
        create_benchmark_config_card(),
        
        # Status Display
        create_status_card(),
        
        # Status interval component
        dcc.Interval(
            id='status-interval',
            interval=1*1000,  # in milliseconds
            n_intervals=0,
            disabled=True  # Initially disabled
        ),
    ])


# Create visualizer instance
runner = BenchmarkRunner()
    
# Set the layout
layout = create_benchmark_layout(runner.get_available_results()) 
    
# Register this page
register_page(__name__, path='/', title="Benchmark", layout=layout)

register_benchmark_callbacks(runner)
register_status_callbacks(runner)
