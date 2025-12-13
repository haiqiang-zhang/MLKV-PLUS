import dash_bootstrap_components as dbc
from dash import html
import os
from visualizer.config_utils import load_config

def create_setting_card():
    """Create a card for process management"""
    return dbc.Card([
        dbc.CardHeader("Setting"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P("Kill all distmlkv processes except the current one:"),
                    dbc.Button(
                        "Kill Processes", 
                        id="kill-processes-button", 
                        color="danger", 
                        className="me-2"
                    ),
                    html.Div(id="kill-processes-result", className="mt-2")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Operation File Directory"),
                    dbc.Input(
                        id="operation-file-dir",
                        type="text",
                        value=load_config()['operation_file_dir'],
                        placeholder="Enter path to operation files directory",
                        persistence=True,
                        persistence_type='local'
                    )
                ], width=12)
            ], className="mt-3")
        ])
    ], className="mb-4")