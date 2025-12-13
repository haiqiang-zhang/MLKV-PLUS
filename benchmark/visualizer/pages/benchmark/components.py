import os
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import cupy as cp
from benchmark.ycsb.ConfigLoader import get_available_workloads
from benchmark.ycsb.binding_registry import get_available_bindings as get_available_bindings_py
from benchmark.ycsb_cpp.python import YCSBBridge

def create_status_card():
    """Create the status display card"""
    return dbc.Card([
        dbc.CardHeader("Benchmark Status"),
        dbc.CardBody([
            html.Div(id="status-display", className="mb-3"),
            dbc.Progress(id="benchmark-progress", value=0, className="mb-3")
        ])
    ], className="mb-4", id="status-card", style={"display": "none"})

def create_binding_config_table():
    """Create an editable table for binding configuration"""
    return html.Div([
        html.Div([
            html.I(className="bi bi-gear-fill me-2"),
            html.Span("Binding Configuration", className="fw-bold fs-6")
        ], className="d-flex align-items-center mb-2"),
        dash_table.DataTable(
            id='binding-config-table',
            columns=[
                {"name": "Parameter", "id": "parameter", "editable": False},
                {"name": "Value", "id": "value", "editable": True}
            ],
            data=[],
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'fontFamily': 'Arial, sans-serif'
            },
            style_header={
                'backgroundColor': '#f8f9fa',
                'fontWeight': 'bold',
                'border': '1px solid #dee2e6'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ]
        )
    ], className="pt-3")

def create_benchmark_config_card():
    """Create the benchmark configuration card"""
    # Get available GPU devices
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_options = [{"label": f"GPU {i}", "value": i} for i in range(num_gpus)]
    except:
        gpu_options = [{"label": "GPU 0", "value": 0}]
    
    # Get available workloads
    workload_options = [{"label": name, "value": name} for name in get_available_workloads()]
    
    # Get available bindings
    binding_options_py = [
        {
            "label": html.Span(
                [
                    html.Img(src="static/assets/python.svg", height=20),
                    html.Span(name, style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'},
            ),
            "value": name + "::python"
        }
        for name in get_available_bindings_py().keys()
    ]
    
    binding_options_cpp = [
        {
            "label": html.Span(
                [
                    html.Img(src="static/assets/cpp.svg", height=20),
                    html.Span(name, style={'font-size': 15, 'padding-left': 10}),
                ], style={'align-items': 'center', 'justify-content': 'center'},
            ),
            "value": name + "::cpp"
        }
        for name in YCSBBridge.get_available_bindings()
    ]
    
    # Combine Python and C++ binding options
    binding_options = binding_options_py + binding_options_cpp
    
    return dbc.Card([
        dbc.CardHeader("Benchmark Configuration"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-cpu me-2"),
                            html.Span("Binding", className="fw-bold fs-5")
                        ], className="d-flex align-items-center mb-2"),
                        dcc.Dropdown(id="binding",
                                   options=binding_options,
                                   value=binding_options[0]['value'] if binding_options else None, 
                                   persistence=True, 
                                   persistence_type='local',
                                   className="border-primary rounded-4"),
                        create_binding_config_table()
                    ], className="p-3 bg-light rounded-4")
                ], width=12, className="mb-4")
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Record Sizes (comma-separated)"),
                    dbc.Input(id="record-sizes", type="text", persistence=True, persistence_type='local')
                ], width=6),
                dbc.Col([
                    dbc.Label("Repeat Count"),
                    dbc.Input(id="repeat-count", type="number", 
                            value=1, min=1, max=10, persistence=True, persistence_type='local')
                ], width=6)
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("GPU Device"),
                    dcc.Dropdown(id="gpu-device", 
                                 options=gpu_options, 
                                 value=[0], 
                                 persistence=True, 
                                 persistence_type='local',
                                 multi=True)
                ], width=6),
                dbc.Col([
                    dbc.Label("Distribution"),
                    dbc.InputGroup([
                        dcc.Dropdown(id="distribution", 
                                   options=[{"label": "Zipfian", "value": "zipfian"},
                                           {"label": "Uniform", "value": "uniform"}],
                                   value="zipfian", persistence=True, persistence_type='local',
                                   style={"flex": "1"}),
                        html.Div([
                            dbc.InputGroupText("Theta"),
                            dbc.Input(id="zipfian-theta", type="number", 
                                    value=0.99, min=0.0, max=1.0, step=0.01,
                                    persistence=True, persistence_type='local',
                                    style={"width": "100px"})
                        ], id="zipfian-theta-container", style={"display": "none"})
                    ])
                ], width=6)
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Number of Streams / Processes (comma-separated)"),
                    dbc.InputGroup([
                        dbc.Input(id="num-streams", type="text",
                                  placeholder="1, 2, 3, 4 .....",
                                  persistence=True, persistence_type='local'),
                        dbc.InputGroupText([
                            dbc.Checkbox(id="unlimited-streams",
                                         persistence=True, persistence_type='local'),
                            "Unlimited"
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Label("Workloads"),
                    dcc.Dropdown(id="workloads", 
                               options=workload_options,
                               value=[workload_options[0]['value']],
                               multi=True,
                               clearable=False,
                               searchable=True,
                               persistence=True,
                               persistence_type='local')
                ], width=6)
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Comments"),
                    dbc.Textarea(
                        id="benchmark-comments",
                        placeholder="Enter comments about this benchmark run...",
                        style={'width': '100%', 'height': 100},
                        persistence=True,
                        persistence_type='local'
                    )
                ], width=12)
            ], className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Operation File"),
                    dbc.InputGroup([
                        dcc.Dropdown(
                            id="operation-file-dropdown",
                            options=[],
                            placeholder="Select operation file",
                            persistence=True,
                            persistence_type='local',
                            style={'flex': '1'}
                        ),
                        dbc.Input(
                            id="operation-file-custom",
                            placeholder="Or enter custom operation file name",
                            persistence=True,
                            persistence_type='local',
                            style={'flex': '1'}
                        )
                    ], style={'display': 'flex', 'gap': '0'})
                ], width=12)
            ], className="mb-2"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Results will be saved to:", className="text-muted me-2")
                            ], width="auto"),
                            dbc.Col([
                                html.Div(id="results-file-path", className="text-primary fw-bold")
                            ], width="auto")
                        ], className="align-items-center")
                    ], className="mt-2")
                ], width=12)
            ], className="mb-2"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Run Benchmarks", id="run-button", color="primary", className="mt-3 me-2")
                ], width="auto"),
                dbc.Col([
                    dbc.Button("Stop Benchmarks", id="stop-button", color="danger", className="mt-3")
                ], width="auto")
            ]),
            
            # placeholder for message
            dbc.Row([
                dbc.Col([
                    html.Div(id="benchmark-config-msg", className="mt-3")
                ], width="auto")
            ])
            
        ])
    ], className="mb-4")