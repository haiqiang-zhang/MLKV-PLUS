import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import io
import base64
from typing import List, Optional
import numpy as np

class MPLLineChart:
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame,
                         x_axis: str,
                         y_axis: str,
                         group_by: Optional[list[str]] = None,
                         x_scale: str = 'linear') -> str:
        """Create a line chart and return it as a base64 string"""
        # Set academic style
        matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define academic color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set x-axis scale
        if x_scale == 'log2':
            ax.set_xscale('log', base=2)
        else:  # linear
            ax.set_xscale('linear')
        
        # Handle no group by or empty group by
        if not group_by:
            # Treat entire dataset as one group
            group_stats = df.groupby(x_axis)[y_axis].agg(['mean', 'std']).reset_index()
            group_stats = group_stats.sort_values(by=x_axis)
            
            # Plot with error bars
            ax.errorbar(group_stats[x_axis], group_stats['mean'],
                       yerr=group_stats['std'],
                       fmt='o-',  # 'o-' means dots connected by lines
                       markersize=8,
                       linewidth=2,
                       color=colors[0],
                       capsize=5,  # Length of error bar caps
                       capthick=2,  # Thickness of error bar caps
                       elinewidth=2,  # Thickness of error bar lines
                       label='All Data')
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
                
                # Plot with error bars
                ax.errorbar(group_stats[x_axis], group_stats['mean'],
                           yerr=group_stats['std'],
                           fmt='o-',  # 'o-' means dots connected by lines
                           markersize=8,
                           linewidth=2,
                           color=colors[idx % len(colors)],
                           capsize=5,  # Length of error bar caps
                           capthick=2,  # Thickness of error bar caps
                           elinewidth=2,  # Thickness of error bar lines
                           label=str(group_value))
        
        # Calculate overall statistics for subtitle
        overall_std = df[y_axis].std()
        
        # Create title with parameters
        title_parts = []
        if 'num_records' in df.columns:
            title_parts.append(f"num records: {df['num_records'].iloc[0]}")
        if 'distribution' in df.columns:
            dist_value = df['distribution'].iloc[0]
            title_parts.append(f"distribution: {dist_value}")
            if dist_value == 'zipfian' and 'zipfian_theta' in df.columns:
                title_parts[-1] += f" | zipfian theta: {df['zipfian_theta'].iloc[0]}"
        if 'orderedinserts' in df.columns:
            title_parts.append(f"orderedinserts: {df['orderedinserts'].iloc[0]}")
        if 'data_integrity' in df.columns:
            title_parts.append(f"data integrity: {df['data_integrity'].iloc[0]}")
        if 'ops' in df.columns:
            title_parts.append(f"ops: {df['ops'].iloc[0]}")
        
        # Set title with parameters
        if title_parts:
            title_text = '\n'.join(title_parts)
            ax.set_title(title_text, fontsize=12, pad=20, loc='left', y=1.1)
        
        # Update layout
        ax.set_xlabel(x_axis, fontsize=14)
        ax.set_ylabel(y_axis, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Adjust figure margins
        plt.subplots_adjust(top=0.85)
        
        return plt
        
    
    @staticmethod
    def get_png_chart(df: pd.DataFrame,
                            x_axis: str,
                            y_axis: str,
                            group_by: list[str],
                            x_scale: str = 'linear') -> str:
        """Create an academic-style chart and return it as a base64 string"""
        plt = MPLLineChart.create_line_chart(df, x_axis, y_axis, group_by, x_scale)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        return img_str

    @staticmethod
    def get_pdf_chart(df: pd.DataFrame,
                        x_axis: str,
                        y_axis: str,
                        group_by: list[str],
                        x_scale: str = 'linear') -> io.BytesIO:
        """Create an academic-style chart and return it as a PDF buffer"""
        plt = MPLLineChart.create_line_chart(df, x_axis, y_axis, group_by, x_scale)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='pdf', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf 