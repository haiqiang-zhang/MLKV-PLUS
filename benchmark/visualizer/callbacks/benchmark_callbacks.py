import threading
from dash import callback, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash import html
from benchmark.ycsb.ConfigLoader import get_available_workloads, get_binding_config
import os
import time
import json
import base64
import subprocess
from ..pages.results.components import create_results_table
from ..BenchmarkRunner import BenchmarkRunner
import dash
from ..config_utils import load_config

def handle_result_file_load(result_file, upload_contents, runner: BenchmarkRunner):
    """Handle loading results from file or upload"""
    if result_file:
        file_path = os.path.join(runner.results_dir, result_file)
        if runner.load_results_from_file(file_path):
            return create_results_table(runner.results, select_all=True)
        return "Error loading results file"
    elif upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            runner.results = json.loads(decoded)
            return create_results_table(runner.results, select_all=True)
        except Exception as e:
            return f"Error processing uploaded file: {str(e)}"
    return "Please select a results file or upload one"

def get_operation_files(directory):
    """Get all operation files from the specified directory"""
    if not directory or not os.path.isdir(directory):
        return []
    
    files = []
    for file in os.listdir(directory):
        if file.endswith('.json'):  # Assuming operation files are JSON files
            files.append({"label": file, "value": file})  # Only return filename as value
    return files

def load_binding_config_data(binding_name):
    """Load binding configuration and format it for DataTable"""
    try:
        # Extract just the binding name (remove ::python or ::cpp suffix)
        clean_binding_name = binding_name.split('::')[0] if binding_name else None
        
        if not clean_binding_name:
            return []
        
        # Load the binding configuration
        config = get_binding_config(clean_binding_name)
        
        # Format data for DataTable
        table_data = []
        for key, value in config.items():
            table_data.append({
                'parameter': key,
                'value': str(value)
            })
        
        return table_data
    
    except Exception as e:
        # Return empty data if binding config not found or error occurs
        print(f"Error loading binding config for {binding_name}: {e}")
        return []

def format_table_data_to_config(table_data):
    """Convert DataTable data back to configuration dictionary"""
    config = {}
    if not table_data:
        return config
        
    for row in table_data:
        param = row.get('parameter', '')
        value = row.get('value', '')
        
        if param and value:
            # Try to convert to appropriate data type
            try:
                # Handle underscore numbers (like 100_000)
                if '_' in value and value.replace('_', '').isdigit():
                    config[param] = int(value.replace('_', ''))
                # Try integer first
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    config[param] = int(value)
                # Try float
                elif '.' in value:
                    try:
                        config[param] = float(value)
                    except ValueError:
                        config[param] = value
                # Keep as string
                else:
                    config[param] = value
            except ValueError:
                # If conversion fails, keep as string
                config[param] = value
    
    return config

def register_benchmark_callbacks(runner: BenchmarkRunner):
    @callback(
        Output("operation-file-dropdown", "options"),
        Input("operation-file-dir", "value"),
        prevent_initial_call=True
    )
    def update_operation_files(directory_values):
        # Get the first non-None value from the list
        directory = next((d for d in directory_values if d is not None), None)
        return get_operation_files(directory)

    @callback(
        Output("binding-config-table", "data"),
        Input("binding", "value")
    )
    def update_binding_config_table(binding_value):
        """Update binding configuration table when binding is selected"""
        if not binding_value:
            return []
        
        return load_binding_config_data(binding_value)

    @callback(
        [Output("operation-file-custom", "disabled"),
         Output("operation-file-dropdown", "disabled")],
        [Input("operation-file-dropdown", "value"),
         Input("operation-file-custom", "value")]
    )
    def toggle_inputs(dropdown_value, custom_value):
        if dropdown_value:
            return True, False  # Disable custom, enable dropdown
        elif custom_value:
            return False, True  # Enable custom, disable dropdown
        return False, False  # Both enabled when empty

    @callback(
        Output("num-streams", "disabled"),
        Input("unlimited-streams", "value")
    )
    def toggle_streams_input(unlimited):
        return unlimited if unlimited else False

    @callback(
        Output("zipfian-theta-container", "style"),
        Input("distribution", "value")
    )
    def toggle_zipfian_theta(distribution):
        if distribution == "zipfian":
            return {"display": "flex"}
        return {"display": "none"}

    @callback(
        Output("benchmark-config-msg", "children"),
        Input("run-button", "n_clicks"),
        [State("record-sizes", "value"),
         State("workloads", "value"),
         State("gpu-device", "value"),
         State("distribution", "value"),
         State("zipfian-theta", "value"),
         State("repeat-count", "value"),
         State("binding", "value"),
         State("benchmark-comments", "value"),
         State("num-streams", "value"),
         State("unlimited-streams", "value"),
         State("operation-file-dropdown", "value"),
         State("operation-file-custom", "value"),
         State("binding-config-table", "data")],
        prevent_initial_call=True
    )
    def start_benchmark(run_clicks, record_sizes, workload_names, gpu_device, distribution, zipfian_theta, repeat_count, binding, comments, num_streams, unlimited_streams, operation_file, custom_operation_file, binding_config_data):
        if not run_clicks:
            return no_update
            
        # Check if benchmark is already running
        if runner.is_benchmark_running():
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "A benchmark is already running. Please wait for it to complete."
            ], color="warning")
            

        record_sizes = [int(x.strip()) for x in record_sizes.split(",")]
        repeat_count = int(repeat_count)
        
        if not runner.check_current_file():
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "No results file selected. Please select a results file or upload one."
            ], color="warning")
            
        # Convert binding configuration table data to config dictionary
        binding_config = format_table_data_to_config(binding_config_data)
        
        # Set num_streams to None if unlimited is checked
        if unlimited_streams:
            num_streams = None
        else:
            num_streams = [int(x.strip()) for x in num_streams.split(",")]
            
        # Construct full path for operation file
        operation_file_path = None
        operation_file_dir = load_config()['operation_file_dir']
        if operation_file_dir:
            if custom_operation_file:
                operation_file_path = os.path.join(operation_file_dir, custom_operation_file)
            elif operation_file:
                operation_file_path = os.path.join(operation_file_dir, operation_file)
            
        runner.run_batch_benchmark(record_sizes=record_sizes, 
                                    workload_names=workload_names, 
                                    distribution=distribution, 
                                    zipfian_theta=zipfian_theta if distribution == "zipfian" else None,
                                    gpu_device=gpu_device, 
                                    repeat_count=repeat_count,
                                    binding_name=binding.split("::")[0],
                                    binding_type=binding.split("::")[1],
                                    comments=comments,
                                    num_streams=num_streams,
                                    operation_file=operation_file_path,
                                    binding_config=binding_config)
            

    @callback(
        Output("workloads", "options"),
        Input("workloads", "value")
    )
    def reload_workloads(value):
        workload_options = [{"label": name, "value": name} for name in get_available_workloads()]
        return workload_options

    @callback(
        Output("benchmark-result-files", "options"),
        Input("benchmark-result-files", "value"),
        prevent_initial_call=False
    )
    def reload_results_file(benchmark_result_file):
        runner.available_results = runner.get_available_results()
        return runner.available_results

    @callback(
        Output("results-file-path", "children"),
        [Input("run-button", "n_clicks"),
         Input("benchmark-result-files", "value")],
        [State("record-sizes", "value"),
         State("workloads", "value"),
         State("gpu-device", "value"),
         State("distribution", "value"),
         State("repeat-count", "value")]
    )
    def update_results_file_path(run_clicks, benchmark_result_file, 
                               record_sizes, workload_names, gpu_device, distribution, repeat_count):
        if benchmark_result_file:
            path = os.path.join(runner.results_dir, benchmark_result_file)
            print(f"Loading results from {path}")
            runner.load_results_from_file(path)
            return path
        else:
            runner.current_file = None
            runner.results = []
            return "No results file selected"

    # @callback(
    #     Output("kill-processes-result", "children"),
    #     Input("kill-processes-button", "n_clicks"),
    #     prevent_initial_call=True
    # )
    # def kill_distmlkv_processes(n_clicks):
    #     if not n_clicks:
    #         return no_update
        
    #     try:
            
            
    #         # Return success message with alert
    #         return dbc.Alert([
    #             html.I(className="bi bi-check-circle-fill me-2"),
    #             "Successfully killed distmlkv processes"
    #         ], color="success", duration=5000)
    #     except Exception as e:
    #         # Return error message with alert
    #         return dbc.Alert([
    #             html.I(className="bi bi-exclamation-triangle-fill me-2"),
    #             f"Error killing processes: {str(e)}"
    #         ], color="danger", duration=5000)

    @callback(
        [Output("benchmark-create-modal", "is_open"),
         Output("benchmark-new-filename", "value"),
         Output("benchmark-result-files", "options", allow_duplicate=True)],
        [Input("benchmark-create-results", "n_clicks"),
         Input("benchmark-cancel-create", "n_clicks"),
         Input("benchmark-confirm-create", "n_clicks")],
        [State("benchmark-create-modal", "is_open"),
         State("benchmark-new-filename", "value")],
        prevent_initial_call=True
    )
    def toggle_create_modal(create_clicks, cancel_clicks, confirm_clicks, is_open, filename):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "benchmark-create-results":
            return True, "", no_update
        elif button_id == "benchmark-cancel-create":
            return False, "", no_update
        elif button_id == "benchmark-confirm-create":
            if filename:
                # Create new results file
                if not filename.endswith('.json'):
                    filename += '.json'
                runner.generate_new_results_file(filename)
                # Refresh the results file list
                runner.available_results = runner.get_available_results()
                return False, "", runner.available_results
            return True, filename, no_update  # Keep modal open if no filename provided
        
        return no_update, no_update, no_update