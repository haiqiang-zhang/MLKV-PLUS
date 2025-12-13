import plotly.graph_objects as go
import pandas as pd
import io
import base64
from typing import List, Optional
import numpy as np

class PlotlyLineChart:
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame,
                         x_axis: str,
                         y_axis: str,
                         group_by: Optional[list[str]] = None,
                         x_scale: str = 'linear') -> go.Figure:
        """Create a line chart and return it as a Plotly figure"""
        # Define academic color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create figure
        fig = go.Figure()
        
        # Handle no group by or empty group by
        if not group_by:
            # Treat entire dataset as one group
            group_stats = df.groupby(x_axis)[y_axis].agg(['mean', 'std']).reset_index()
            group_stats = group_stats.sort_values(by=x_axis)
            
            # Add trace with error bars
            fig.add_trace(go.Scatter(
                x=group_stats[x_axis],
                y=group_stats['mean'],
                error_y=dict(
                    type='data',
                    array=group_stats['std'],
                    visible=True
                ),
                mode='lines+markers',
                name='All Data',
                line=dict(color=colors[0], width=2),
                marker=dict(size=8, color=colors[0])
            ))
        else:
            # Handle multiple group by columns
            if isinstance(group_by, list) and len(group_by) > 1:
                df['combined_group'] = df[group_by].apply(lambda x: ' | '.join(x.astype(str)), axis=1)
                group_column = 'combined_group'
            else:
                group_column = group_by[0] if isinstance(group_by, list) else group_by
            
            # Plot data with error bars for each group
            for idx, group_value in enumerate(df[group_column].unique()):
                group_data = df[df[group_column] == group_value]
                # Calculate mean and std for each x value within the group
                group_stats = group_data.groupby(x_axis)[y_axis].agg(['mean', 'std']).reset_index()
                group_stats = group_stats.sort_values(by=x_axis)
                
                # Add trace with error bars
                fig.add_trace(go.Scatter(
                    x=group_stats[x_axis],
                    y=group_stats['mean'],
                    error_y=dict(
                        type='data',
                        array=group_stats['std'],
                        visible=True
                    ),
                    mode='lines+markers',
                    name=str(group_value),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=8, color=colors[idx % len(colors)])
                ))
        
        # Create title with parameters
        title_columns = ['num_records', 'distribution', 'orderedinserts', 'data_integrity', 'num_batch_ops', 'num_streams', 'workload', 'init_capacity', 'max_field_length', 'min_field_length']
        title_parts = []
        for col in title_columns:
            if col in df.columns and col != x_axis and col not in group_by and not (x_axis == 'multiget_batch_size' and col == 'workload'):
                if col == 'distribution':
                    dist_value = df['distribution'].iloc[0]
                    title_parts.append(f"distribution: {dist_value}")
                    if dist_value == 'zipfian' and 'zipfian_theta' in df.columns:
                        title_parts[-1] += f" | zipfian theta: {df['zipfian_theta'].iloc[0]}"
                else:
                    value = df[col].unique()
                    title_parts.append(f"{col}: {', '.join(str(v) for v in value)}")
        # if 'num_records' in df.columns:
        #     title_parts.append(f"num records: {df['num_records'].iloc[0]}")
        # if 'distribution' in df.columns:
        #     dist_value = df['distribution'].iloc[0]
        #     title_parts.append(f"distribution: {dist_value}")
        #     if dist_value == 'zipfian' and 'zipfian_theta' in df.columns:
        #         title_parts[-1] += f" | zipfian theta: {df['zipfian_theta'].iloc[0]}"
        # if 'orderedinserts' in df.columns:
        #     title_parts.append(f"orderedinserts: {df['orderedinserts'].iloc[0]}")
        # if 'data_integrity' in df.columns:
        #     # list all unique data integrity values
        #     data_integrity_values = df['data_integrity'].unique()
        #     title_parts.append(f"data integrity: {', '.join(str(di) for di in data_integrity_values)}")
        # if 'num_batch_ops' in df.columns:
        #     title_parts.append(f"num batch ops: {df['num_batch_ops'].iloc[0]}")
        # if 'num_streams' in df.columns:
        #     num_streams_values = df['num_streams'].unique()
        #     title_parts.append(f"num streams: {', '.join(str(s) for s in num_streams_values)}")
        # if 'workload' in df.columns:
        #     workload_values = df['workload'].unique()
        #     title_parts.append(f"workload: {', '.join(str(w) for w in workload_values)}")
        
        # Update layout
        fig.update_layout(
            title='<br>'.join(title_parts) if title_parts else None,
            title_font=dict(size=12),
            title_x=0.02,  # Align title to the left
            title_y=0.98,  # Position title at the top
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            font=dict(size=14),
            # Remove fixed height to allow responsive sizing
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            margin=dict(l=60, r=120, t=120, b=60),  # Better margins for labels
            template="plotly_white"
        )
        
        # Set x-axis scale
        if x_scale == 'log2':
            fig.update_xaxes(type='log', dtick=1)  # dtick=1 means ticks at 2^n
        else:  # linear
            fig.update_xaxes(type='linear')
        
        return fig
    
    @staticmethod
    def get_chart(df: pd.DataFrame,
                     x_axis: str,
                     y_axis: str,
                     group_by: list[str],
                     x_scale: str = 'linear') -> str:
        """Create a chart and return it as a base64 string"""
        fig = PlotlyLineChart.create_line_chart(df, x_axis, y_axis, group_by, x_scale)
    
        return fig

    @staticmethod
    def get_pdf_chart(df: pd.DataFrame,
                     x_axis: str,
                     y_axis: str,
                     group_by: list[str],
                     x_scale: str = 'linear') -> io.BytesIO:
        """Create a chart and return it as a PDF buffer"""
        fig = PlotlyLineChart.create_line_chart(df, x_axis, y_axis, group_by, x_scale)
        
        buf = io.BytesIO()
        fig.write_image(buf, format='pdf')
        buf.seek(0)
        
        return buf 