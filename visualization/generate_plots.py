"""
Visualization module for PromptPressure evaluation results.
Generates plots and summary statistics from evaluation data.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure plotting style
plt.style.use('seaborn')
sns.set_palette('viridis')

class ResultVisualizer:
    def __init__(self, results_dir: str = 'outputs', output_dir: str = 'visualization'):
        """Initialize the visualizer with directories for input and output."""
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def load_results(self) -> pd.DataFrame:
        """Load all CSV results from the results directory."""
        result_files = list(self.results_dir.glob('*.csv'))
        if not result_files:
            raise FileNotFoundError(f"No result files found in {self.results_dir}")
            
        dfs = []
        for file in result_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = file.name
                df['evaluation_time'] = datetime.fromtimestamp(file.stat().st_mtime)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
                
        if not dfs:
            raise ValueError("No valid result files could be loaded")
            
        return pd.concat(dfs, ignore_index=True)
    
    def plot_success_rate(self, df: pd.DataFrame) -> None:
        """Plot success rate over time by model."""
        plt.figure(figsize=(12, 6))
        
        # Calculate success rate
        df['success'] = df['status'] == 'success'
        success_rates = df.groupby(['model_name', pd.Grouper(key='evaluation_time', freq='D')])['success'] \
            .mean() \
            .reset_index()
            
        # Plot
        sns.lineplot(
            data=success_rates,
            x='evaluation_time',
            y='success',
            hue='model_name',
            marker='o'
        )
        
        plt.title('Model Success Rate Over Time')
        plt.xlabel('Date')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.1)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rate.png')
        plt.close()
    
    def plot_latency_distribution(self, df: pd.DataFrame) -> None:
        """Plot latency distribution by model."""
        if 'latency' not in df.columns:
            return
            
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df[df['status'] == 'success'],
            x='model_name',
            y='latency',
            showfliers=False
        )
        
        plt.title('Latency Distribution by Model')
        plt.xlabel('Model')
        plt.ylabel('Latency (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png')
        plt.close()
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from the results."""
        stats = {
            'total_evaluations': len(df),
            'models_tested': df['model_name'].nunique(),
            'success_rate': df['status'].eq('success').mean(),
            'by_model': {}
        }
        
        # Model-specific stats
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            stats['by_model'][model] = {
                'count': len(model_df),
                'success_rate': model_df['status'].eq('success').mean(),
                'avg_latency': model_df.get('latency', pd.Series([0])).mean(),
                'last_evaluated': model_df['evaluation_time'].max().isoformat()
            }
            
        return stats
    
    def save_summary(self, stats: Dict[str, Any]) -> None:
        """Save summary statistics to a JSON file."""
        with open(self.output_dir / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def run(self) -> None:
        """Run all visualizations and generate reports."""
        try:
            df = self.load_results()
            
            # Generate visualizations
            self.plot_success_rate(df)
            self.plot_latency_distribution(df)
            
            # Generate and save summary
            stats = self.generate_summary_stats(df)
            self.save_summary(stats)
            
            print(f"Visualizations saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            raise

if __name__ == "__main__":
    visualizer = ResultVisualizer()
    visualizer.run()
