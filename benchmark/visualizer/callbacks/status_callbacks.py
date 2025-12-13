import time
from dash import callback, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
from dash import html

def register_status_callbacks(runner):
    @callback(
        [Output("status-display", "children"),
         Output("benchmark-progress", "value"),
         Output("status-card", "style"),
         Output("run-button", "style")],
        Input("status-interval", "n_intervals")
    )
    def update_status(n_intervals):
        status = runner.current_status
        if status and status.get('running', False):
            progress = ((status['current_combination']-1)*status['repeat_count']+status['current_repeat']-1) / (status['total_combinations']*status['repeat_count']) * 100
            current_record = status['current_record_size']
            current_workload = status['current_workload']
            
            status_display = dbc.Alert([
                html.I(className="bi bi-play-circle-fill me-2"),
                "Benchmark is running",
                html.Br(),
                f"Test Set Progress: {status['current_combination']}/{status['total_combinations']}",
                html.Br(),
                f"Current record size: {current_record:,} records",
                html.Br(),
                f"Current workload: {current_workload}",
                html.Br(),
                f"Current num stream: {status['current_num_stream']}",
                html.Br(),
                f"Repeat: {status.get('current_repeat', 0)}/{status.get('repeat_count', 1)}",
                html.Br(),
                f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['start_time']))}"
            ], color="info")
            
            return status_display, progress, {"display": "block"}, {"display": "none"}
            
        elif status and not status.get('running', False):
            if 'error' in status:
                status_display = dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    "Benchmark completed with error",
                    html.Br(),
                    f"Error: {status['error']}",
                    html.Br(),
                    f"Duration: {status['duration']:.2f} seconds"
                ], color="danger")
            elif status.get('stopped', False):
                status_display = no_update
            else:
                status_display = dbc.Alert([
                    html.I(className="bi bi-check-circle-fill me-2"),
                    "Benchmark completed successfully",
                    html.Br(),
                    f"Duration: {status['duration']:.2f} seconds"
                ], color="success")
            
            return status_display, 100, {"display": "block"}, {"display": "block"}
        
        return None, None, {"display": "none"}, {"display": "block"}

    @callback(
        Output("status-interval", "disabled"),
        [Input("run-button", "n_clicks"),
         Input("status-interval", "n_intervals")],
        [State("status-interval", "disabled")]
    )
    def toggle_status_interval(run_clicks, n_intervals, current_disabled):
        ctx = callback_context
        if not ctx.triggered:
            status = runner.current_status
            if status and status.get('running', False):
                return False
            return True
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "run-button" and run_clicks:
            return False
            
        if trigger_id == "status-interval":
            status = runner.current_status
            if status and status.get('running', False):
                return False
            return True
            
        return current_disabled 
    
    
    @callback(
        [Output("status-display", "children", allow_duplicate=True),
         Output("status-card", "style", allow_duplicate=True)],
        Input("stop-button", "n_clicks"),
        prevent_initial_call=True
    )
    def stop_benchmark(stop_clicks):
        if not stop_clicks:
            return no_update, no_update
            
        if not runner.is_benchmark_running():
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                "No benchmark is currently running."
            ], color="warning"), {"display": "block"}
            
        try:
            runner.stop_benchmark()
            return dbc.Alert([
                html.I(className="bi bi-stop-circle-fill me-2"),
                "Benchmark stopped successfully."
            ], color="success"), {"display": "block"}
        except Exception as e:
            return dbc.Alert([
                html.I(className="bi bi-exclamation-triangle-fill me-2"),
                f"Error stopping benchmark: {str(e)}"
            ], color="danger"), {"display": "block"}