import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import yaml
from scipy import stats

class PerformanceAnalyzer:
    def __init__(self, simulation_results_dir='experiment_lab/rl_experiment_runs', output_dir='output/performance_analysis'):
        self.simulation_results_dir = simulation_results_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.all_simulation_data = self._load_all_simulation_data()

    def _load_all_simulation_data(self):
        all_dfs = []
        for agent_folder in os.listdir(self.simulation_results_dir):
            agent_path = os.path.join(self.simulation_results_dir, agent_folder)
            if os.path.isdir(agent_path):
                for file_name in os.listdir(agent_path):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(agent_path, file_name)
                        try:
                            df = pd.read_csv(file_path)
                            df['agent_id'] = agent_folder # Assume folder name is agent ID
                            all_dfs.append(df)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def analyze_kpis(self, kpis_to_analyze=['num_delivered_orders', 'total_driver_distance', 'total_driver_time']):
        if self.all_simulation_data.empty:
            print("No simulation data loaded for analysis.")
            return

        summary = self.all_simulation_data.groupby('agent_id')[kpis_to_analyze].agg(['mean', 'std', 'min', 'max'])
        print("\n--- KPI Summary per Agent ---")
        print(summary)
        summary.to_csv(os.path.join(self.output_dir, "kpi_summary.csv"))

        # Visualize KPI comparison
        for kpi in kpis_to_analyze:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='agent_id', y=kpi, data=self.all_simulation_data, errorbar='sd', capsize=.2)
            plt.title(f'Comparison of {kpi} across Agents')
            plt.ylabel(kpi)
            plt.xlabel('Agent ID')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{kpi}_comparison.png"))
            plt.close()

    def analyze_time_series(self, metrics_to_plot=['num_delivered_orders', 'num_pending_orders', 'total_driver_distance']):
        if self.all_simulation_data.empty:
            print("No simulation data loaded for time series analysis.")
            return

        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 7))
            sns.lineplot(x='time', y=metric, hue='agent_id', data=self.all_simulation_data, errorbar='sd')
            plt.title(f'{metric} Over Simulation Time')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel(metric)
            plt.legend(title='Agent')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{metric}_time_series.png"))
            plt.close()

    def conduct_statistical_tests(self, kpi='num_delivered_orders'):
        if self.all_simulation_data.empty:
            print("No data for statistical tests.")
            return

        agents = self.all_simulation_data['agent_id'].unique()
        if len(agents) < 2:
            print("Need at least two agents for comparison.")
            return

        print(f"\n--- Statistical Tests for {kpi} ---")
        base_agent = agents[0]
        base_data = self.all_simulation_data[self.all_simulation_data['agent_id'] == base_agent][kpi]

        for i in range(1, len(agents)):
            other_agent = agents[i]
            other_data = self.all_simulation_data[self.all_simulation_data['agent_id'] == other_agent][kpi]

            # Assuming independent samples, use Welch's t-test (handles unequal variances)
            t_stat, p_value = stats.ttest_ind(base_data, other_data, equal_var=False)
            
            print(f"Comparing '{base_agent}' vs '{other_agent}' on '{kpi}':")
            print(f"  Welch's t-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.3f}")
            if p_value < 0.05:
                print(f"  Result: Statistically significant difference (p < 0.05).")
            else:
                print(f"  Result: No significant difference (p >= 0.05).")

    def generate_full_report(self):
        report_path = os.path.join(self.output_dir, f"performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                img {{ max-width: 80%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; }}
                pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>Logistics System Performance Analysis Report</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>KPI Summary</h2>
            <pre>{self.all_simulation_data.groupby('agent_id').agg(['mean', 'std']).to_string()}</pre>
            <img src="num_delivered_orders_comparison.png" alt="Delivered Orders Comparison">
            <img src="total_driver_distance_comparison.png" alt="Driver Distance Comparison">
            
            <h2>Time Series Analysis</h2>
            <img src="num_delivered_orders_time_series.png" alt="Delivered Orders Time Series">
            <img src="total_driver_distance_time_series.png" alt="Driver Distance Time Series">

            <h2>Statistical Significance (Example: Delivered Orders)</h2>
            <pre>Check console output for detailed statistical test results.</pre>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"\nFull report generated at: {report_path}")

if __name__ == '__main__':
    # Ensure dummy experiment data exists for demonstration
    sim_dir = 'experiment_lab/rl_experiment_runs'
    agent_a_dir = os.path.join(sim_dir, 'agent_A')
    agent_b_dir = os.path.join(sim_dir, 'agent_B')
    os.makedirs(agent_a_dir, exist_ok=True)
    os.makedirs(agent_b_dir, exist_ok=True)

    # Generate dummy data for agent A
    for i in range(5):
        time_points = np.arange(0, 600, 30)
        data = {
            'time': time_points,
            'num_delivered_orders': np.cumsum(np.random.randint(1, 5, size=len(time_points))) + np.random.normal(0, 2),
            'total_driver_distance': np.cumsum(np.random.uniform(100, 500, size=len(time_points))) + np.random.normal(0, 100),
            'total_driver_time': np.cumsum(np.random.uniform(30, 90, size=len(time_points))) + np.random.normal(0, 50)
        }
        pd.DataFrame(data).to_csv(os.path.join(agent_a_dir, f'run_{i+1}.csv'), index=False)

    # Generate dummy data for agent B (slightly better performance)
    for i in range(5):
        time_points = np.arange(0, 600, 30)
        data = {
            'time': time_points,
            'num_delivered_orders': np.cumsum(np.random.randint(2, 6, size=len(time_points))) + np.random.normal(0, 2) + 5, # Better performance
            'total_driver_distance': np.cumsum(np.random.uniform(90, 450, size=len(time_points))) + np.random.normal(0, 100), # Slightly less distance
            'total_driver_time': np.cumsum(np.random.uniform(25, 80, size=len(time_points))) + np.random.normal(0, 50) # Slightly less time
        }
        pd.DataFrame(data).to_csv(os.path.join(agent_b_dir, f'run_{i+1}.csv'), index=False)

    analyzer = PerformanceAnalyzer()
    analyzer.analyze_kpis()
    analyzer.analyze_time_series()
    analyzer.conduct_statistical_tests(kpi='num_delivered_orders')
    analyzer.generate_full_report()