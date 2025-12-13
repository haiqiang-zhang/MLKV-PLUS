import json
import base64
import pandas as pd
import time
from dash import callback, Input, Output, State, callback_context, ALL, no_update
import dash_bootstrap_components as dbc
from dash import html
import os
from ..pages.results.components import create_results_table
from ..BenchmarkRunner import BenchmarkRunner

def handle_result_file_load(result_file, upload_contents, runner):
    """Handle loading results from file or upload"""
    if result_file:
        file_path = os.path.join(runner.results_dir, result_file)
        if runner.load_results_from_file(file_path):
            return runner.results
        return "Error loading results file"
    elif upload_contents:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            runner.results = json.loads(decoded)
            return runner.results
        except Exception as e:
            return f"Error processing uploaded file: {str(e)}"
    return "Please select a results file or upload one"


def get_results_columns(runner):
    df = pd.DataFrame(runner.results)
    if 'operation_details' in df.columns:
        df = df.drop(columns=['operation_details'])
    return df.columns

def save_changes(runner, display_columns):
    """Save current results to JSON file"""
    if not runner.current_file:
        return html.Div("No results file loaded", className="alert alert-warning")
        
    try:
        with open(runner.current_file, 'r') as f:
            current_content = json.load(f)
        
        current_json = json.dumps(current_content, sort_keys=True)
        new_json = json.dumps(runner.results, sort_keys=True)
        
        if current_json == new_json:
            return html.Div([
                html.Div("No changes detected, nothing to save", className="alert alert-info"),
                create_results_table(runner.results, select_all=True, display_columns=display_columns)
            ])
            
        with open(runner.current_file, 'w') as f:
            json.dump(runner.results, f, indent=4)
            return html.Div([
                html.Div("Changes saved successfully", className="alert alert-success"),
                create_results_table(runner.results, select_all=True, display_columns=display_columns)
            ])
    except Exception as e:
        return html.Div([
            html.Div(f"Error saving changes: {str(e)}", className="alert alert-danger"),
            create_results_table(pd.DataFrame(runner.results), select_all=True, display_columns=display_columns)
        ])

def register_results_callbacks(runner: BenchmarkRunner):
    @callback(
        Output("results-table", "children"),
        [Input("results-result-files", "value"),
         Input("results-upload-results", "contents"),
         Input("display-columns", "value")]
    )
    def update_results_table(results_result_file, results_upload_contents, display_columns):
        if results_result_file or results_upload_contents:
            handle_result_file_load(results_result_file, results_upload_contents, runner)
        else:
            return html.Div("No results available")
    
        if not display_columns:
            return html.Div("No columns selected")
        else:
            df = pd.DataFrame(runner.results)
            return create_results_table(df, select_all=True, display_columns=display_columns)
    
    
    @callback(
        [Output("display-columns", "options"),
         Output("display-columns", "value")],
        [Input("results-result-files", "value"),
         Input("results-upload-results", "contents")],
        [State("display-columns", "value"),
         State("display-columns-store", "data")]
    )
    def update_display_columns_options(results_result_file, results_upload_contents, current_columns, stored_columns):
        if not runner.results:
            no_results = True
            if results_result_file or results_upload_contents:
                handle_result_file_load(results_result_file, results_upload_contents, runner)
                no_results = False
            if no_results:
                return [], stored_columns or []
        
        columns = get_results_columns(runner)
        options = [{"label": col, "value": col} for col in columns]
        
        # Use stored columns if available, otherwise use current columns
        selected_columns = stored_columns or current_columns or []
        
        # Filter to only include valid columns
        valid_columns = [col for col in selected_columns if col in columns]
        if valid_columns:
            return options, valid_columns
        
        # If no valid columns, return all columns
        return options, columns

    @callback(
        Output("display-columns-store", "data"),
        Input("display-columns", "value"),
        prevent_initial_call=True
    )
    def store_display_columns(selected_columns):
        if selected_columns:
            return selected_columns
        return no_update
    
    @callback(
        Output("results-table", "children", allow_duplicate=True),
        [Input("save-changes-button", "n_clicks")],
        [State("display-columns", "value")],
        prevent_initial_call=True
    )
    def save_results_changes(n_clicks, display_columns):
        if n_clicks:
            return save_changes(runner, display_columns)
        return no_update
    
    @callback(
        Output({"type": "row-select", "index": ALL}, "value"),
        [Input("select-all-button", "n_clicks"),
         Input("clear-all-button", "n_clicks")],
        [State({"type": "row-select", "index": ALL}, "id")]
    )
    def select_all_rows(select_clicks, clear_clicks, row_ids):
        ctx = callback_context
        if not ctx.triggered:
            return [no_update] * len(row_ids)
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "select-all-button":
            return [["selected"] for _ in row_ids]
        elif trigger_id == "clear-all-button":
            return [[] for _ in row_ids]
            
        return [no_update] * len(row_ids)
    
    @callback(
        Output("selected-results", "data"),
        [Input({"type": "row-select", "index": ALL}, "value")],
        [State("results-table", "children")]
    )
    def update_selected_results(selected_values, results_table):
        if not results_table or isinstance(results_table, str):
            return []
            
        selected_indices = []
        for i, value in enumerate(selected_values):
            if value and "selected" in value:
                selected_indices.append(i)
                
        return selected_indices
    
    @callback(
        Output("results-result-files", "options"),
        Input("results-result-files", "value"),
        prevent_initial_call=False
    )
    def reload_results_file(results_result_file):
        runner.available_results = runner.get_available_results()
        return runner.available_results

    @callback(
        Output("results-table", "children", allow_duplicate=True),
        [Input({"type": "delete-row", "index": ALL}, "n_clicks")],
        [State("display-columns", "value"),
         State({"type": "delete-row", "index": ALL}, "id")],
        prevent_initial_call=True
    )
    def handle_delete_row(delete_clicks, display_columns, delete_ids):
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update
            
        # Get the index of the row to delete
        trigger_id = ctx.triggered[0]['prop_id']
        if not trigger_id or "delete-row" not in trigger_id:
            return no_update
            
        # Find the index of the clicked button
        clicked_index = None
        for i, (click, delete_id) in enumerate(zip(delete_clicks, delete_ids)):
            if click and click > 0:  # Only consider actual clicks
                clicked_index = delete_id['index']
                break
                
        if clicked_index is not None:
            # Delete the row
            runner.results.pop(clicked_index)
            
            # Update the table with remaining rows
            if runner.results:
                df = pd.DataFrame(runner.results)
                return create_results_table(df, select_all=True, display_columns=display_columns)
            else:
                return html.Div("No results available")
            
        return no_update

