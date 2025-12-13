import pandas as pd
from dash import callback, Input, Output, State, no_update, dcc, callback_context
from dash import html
import time
import json
import base64
import os
import importlib
from ..BenchmarkRunner import BenchmarkRunner


def results_to_df(results, selected_indices=None):
    df = pd.DataFrame(results)
    if 'operation_details' in df.columns:
        df = df.drop(columns=['operation_details'])
    if selected_indices is not None:
        df = df.iloc[selected_indices]
    return df
    

def handle_result_file_load(result_file, upload_contents, runner):
    """Handle loading results from file or upload"""
    if result_file:
        file_path = os.path.join(runner.results_dir, result_file)
        if runner.load_results_from_file(file_path):
            return runner.results
        return None
    elif upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            runner.results = json.loads(decoded)
            return runner.results
        except Exception as e:
            print(f"Error processing uploaded file: {str(e)}")
            return None
    return None

def get_chart_class(chart_type):
    """Dynamically import and return the chart class for the given type"""
    try:
        module = importlib.import_module(f'visualizer.charts.{chart_type}')
        for name, obj in module.__dict__.items():
            if name.endswith('Chart'):
                return obj
        raise ImportError(f"No chart class found in {chart_type}")
    except Exception as e:
        print(f"Error loading chart class {chart_type}: {str(e)}")
        return None

def register_visualization_callbacks(runner: BenchmarkRunner):
    @callback(
        [Output("x-axis-dropdown", "options"),
         Output("y-axis-dropdown", "options"),
         Output("group-by-dropdown", "options")],
        [Input("results-table", "children"),
         Input("results-result-files", "value"),
         Input("results-upload-results", "contents")]
    )
    def update_visualization_options(results_table, results_result_file, results_upload_contents):
        if not runner.results:
            # Try to load results from file if available
            if results_result_file or results_upload_contents:
                handle_result_file_load(results_result_file, results_upload_contents, runner)
            else:
                return [], [], []
            
        df = results_to_df(runner.results)
        
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object','int64']).columns.tolist()
        
        x_options = [{"label": col, "value": col} for col in numeric_columns]
        y_options = [{"label": col, "value": col} for col in numeric_columns]
        group_options = [{"label": col, "value": col} for col in categorical_columns]
        
        # if not x_axis:
        #     x_axis = "multiget_batch_size"
        # if not y_axis:
        #     y_axis = "throughput"
        # if not group_by:
        #     group_by = "num_streams"
            
        # return x_options, y_options, group_options, x_axis, y_axis, group_by
        return x_options, y_options, group_options

    @callback(
        Output("chart-container", "children"),
        [Input("x-axis-dropdown", "value"),
         Input("y-axis-dropdown", "value"),
         Input("group-by-dropdown", "value"),
         Input("visualization-type", "value"),
         Input("x-axis-scale", "value"),
         Input("selected-results", "data"),
         Input("results-result-files", "value"),
         Input("results-upload-results", "contents")],
        [State("chart-container", "children")],
        prevent_initial_call=True
    )
    def update_chart(x_axis, y_axis, group_by, viz_type, x_scale, selected_indices, 
                    results_result_file, results_upload_contents, current_chart):
        if not runner.results:
            # Try to load results from file if available
            if results_result_file or results_upload_contents:
                results = handle_result_file_load(results_result_file, results_upload_contents, runner)
                if results:
                    runner.results = results
                else:
                    return html.Div([
                        html.I(className="bi bi-graph-up fs-1 mb-3 text-muted"),
                        html.H4("No Chart Available", className="mb-2"),
                        html.P("Select visualization options to generate a visualization", 
                              className="text-muted")
                    ], className="text-center py-5")
            else:
                return html.Div([
                    html.I(className="bi bi-graph-up fs-1 mb-3 text-muted"),
                    html.H4("No Chart Available", className="mb-2"),
                    html.P("Select visualization options to generate a visualization", 
                          className="text-muted")
                ], className="text-center py-5")
            
        if not all([x_axis, y_axis, viz_type]):
            return html.Div([
                html.I(className="bi bi-graph-up fs-1 mb-3 text-muted"),
                html.H4("No Chart Available", className="mb-2"),
                html.P("Select visualization options to generate a visualization", 
                      className="text-muted")
            ], className="text-center py-5")
            
        try:
            df = results_to_df(runner.results, selected_indices)
            chart_class = get_chart_class(viz_type)
            
            if not chart_class:
                raise ImportError(f"Could not load chart class for {viz_type}")
            
            if hasattr(chart_class, 'get_chart'):
                fig = chart_class.get_chart(df, x_axis, y_axis, group_by, x_scale)
                config = {
                        'toImageButtonOptions': { 
                            'height': None, 
                            'width': None, 
                            'format': 'png',
                            'scale': 15
                            },
                        'displaylogo': False
                        }
                return html.Div([
                    dcc.Graph(figure=fig, style={'height': '70vh', 'width': '100%'}, config=config)
                ])
            elif hasattr(chart_class, 'get_png_chart'):
                img_str = chart_class.get_png_chart(df, x_axis, y_axis, group_by, x_scale)
                return html.Div([
                    html.Img(src=f'data:image/png;base64,{img_str}', 
                            className="img-fluid",
                            style={'max-width': '100%', 'height': 'auto'})
                ])
            else:
                raise AttributeError(f"Chart class {viz_type} does not have required methods")
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="bi bi-exclamation-triangle fs-1 mb-3 text-danger"),
                    html.H4("Error Generating Chart", className="mb-2 text-danger"),
                    html.P(str(e), className="text-muted")
                ], className="text-center py-5")
            ])

    @callback(
        Output("download-pdf", "data"),
        Input("save-pdf", "n_clicks"),
        [State("x-axis-dropdown", "value"),
         State("y-axis-dropdown", "value"),
         State("group-by-dropdown", "value"),
         State("visualization-type", "value"),
         State("selected-results", "data")],
        prevent_initial_call=True
    )
    def save_as_pdf(n_clicks, x_axis, y_axis, group_by, viz_type, selected_indices):
        if not n_clicks or not runner.results or not all([x_axis, y_axis, group_by, viz_type]):
            return None
            
        try:
            df = results_to_df(runner.results, selected_indices)
            chart_class = get_chart_class(viz_type)
            
            if not chart_class:
                raise ImportError(f"Could not load chart class for {viz_type}")
            
            if not hasattr(chart_class, 'get_pdf_chart'):
                raise AttributeError(f"Chart class {viz_type} does not support PDF export")
            
            buf = chart_class.get_pdf_chart(df, x_axis, y_axis, group_by)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_chart_{timestamp}.pdf"
            
            return dcc.send_bytes(buf.getvalue(), filename)
            
        except Exception as e:
            print(f"Error saving PDF: {str(e)}")
            return None 