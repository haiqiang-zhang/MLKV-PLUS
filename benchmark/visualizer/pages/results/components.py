import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd
import os
import importlib
import inspect

def get_available_chart_types():
    """Scan the charts directory and return available chart types"""
    chart_types = []
    charts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'charts')
    
    for file in os.listdir(charts_dir):
        if file.endswith('.py') and not file.startswith('__'):
            module_name = file[:-3]  # Remove .py extension
            try:
                module = importlib.import_module(f'visualizer.charts.{module_name}')
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.endswith('Chart'):
                        # Extract the prefix (e.g., 'MPL' from 'MPLLineChart')
                        prefix = name[:-5]  # Remove 'Chart' suffix
                        chart_types.append({
                            "label": f"{prefix} ({module_name})",
                            "value": module_name
                        })
            except Exception as e:
                print(f"Error loading chart module {module_name}: {str(e)}")
    
    return chart_types


def create_results_card():
    """Create the results display card"""
    return dbc.Card([
        dbc.CardHeader("Results"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Save Changes", 
                              id="save-changes-button", 
                              color="success",
                              className="me-2"),
                    dbc.Button("Select All", 
                              id="select-all-button", 
                              color="primary",
                              className="me-2"),
                    dbc.Button("Clear All", 
                              id="clear-all-button", 
                              color="secondary")
                ], width=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Display Columns"),
                    dcc.Dropdown(
                        id="display-columns",
                        multi=True,
                        clearable=False,
                        searchable=True
                    ),
                    dcc.Store(id='display-columns-store', storage_type='local')
                ], width=12)
            ], className="mb-3"),
            html.Div(id="results-table")
        ])
    ], className="mb-4")

def create_visualization_config_card():
    """Create the visualization configuration card"""
    chart_types = get_available_chart_types()
    return dbc.Card([
        dbc.CardHeader("Visualization Configuration"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Visualization Type"),
                    dcc.Dropdown(
                        id="visualization-type",
                        options=chart_types,
                        value=chart_types[0]['value'] if chart_types else None
                    ),
                    dcc.Store(id='visualization-type-store', storage_type='local')
                ], width=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("X-Axis"),
                    dcc.Dropdown(id="x-axis-dropdown",
                                 searchable=True,
                                 persistence=True,
                                 persistence_type='local')
                ], width=4),
                dbc.Col([
                    html.Label("Y-Axis"),
                    dcc.Dropdown(id="y-axis-dropdown",
                                 searchable=True,
                                 persistence=True,
                                 persistence_type='local')
                ], width=4),
                dbc.Col([
                    html.Label("Group By"),
                    dcc.Dropdown(id="group-by-dropdown",
                                 multi=True,
                                 searchable=True,
                                 persistence=True,
                                 persistence_type='local')
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("X-Axis Scale"),
                    dcc.Dropdown(
                        id="x-axis-scale",
                        options=[
                            {"label": "Linear", "value": "linear"},
                            {"label": "Log2", "value": "log2"}
                        ],
                        value="linear",
                        persistence=True,
                        persistence_type='local'
                    )
                ], width=4)
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Download(id="download-pdf"),
                    dbc.Button("Save as PDF", id="save-pdf", color="success", className="mt-3")
                ], width=12)
            ])
        ])
    ], className="mb-4")

def create_chart_card():
    """Create the chart display card"""
    return dbc.Card([
        dbc.CardHeader("Visualization"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className="bi bi-graph-up fs-1 mb-3 text-muted"),
                    html.H4("No Chart Available", className="mb-2"),
                    html.P("Select visualization options to generate a visualization", 
                          className="text-muted")
                ], className="text-center py-5")
            ], id="chart-container")
        ])
    ])

def create_results_table(results, select_all=True, display_columns=None):
    """Create a table with delete buttons and checkboxes for each row"""
    if isinstance(results, pd.DataFrame):
        df = results
    else:
        df = pd.DataFrame(results)
        
    if 'operation_details' in df.columns:
        df = df.drop(columns=['operation_details'])
    
    if display_columns:
        df = df[display_columns]
        
    header = html.Thead(html.Tr([
        html.Th("Select"),
        html.Th("Actions")
    ] + [html.Th(col) for col in df.columns]))
    
    body = html.Tbody([
        html.Tr([
            html.Td(
                dcc.Checklist(
                    id={"type": "row-select", "index": i},
                    options=[{"label": "", "value": "selected"}],
                    value=["selected"] if select_all else [],
                    className="mb-0"
                )
            ),
            html.Td(
                dbc.Button(
                    "Delete",
                    id={"type": "delete-row", "index": i},
                    color="danger",
                    size="sm"
                )
            )
        ] + [
            html.Td(str(df.iloc[i][col])) for col in df.columns
        ]) for i in range(len(df))
    ])
    
    return dbc.Table([header, body], striped=True, bordered=True, hover=True) 


