import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import random
import pickle

class TehranFleetEnvironmentGenerator:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.graph = self._load_graph()
        self.gdf_nodes, self.gdf_edges = ox.graph_to_gdfs(self.graph)

    def _load_config(self, config_path):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_graph(self):
        graph_path = self.config['graph_serialization']['output_path']
        return ox.load_graphml(graph_path)

    def generate_drivers(self, num_drivers):
        driver_nodes = random.sample(list(self.graph.nodes()), num_drivers)
        drivers = []
        for i, node in enumerate(driver_nodes):
            drivers.append({
                'driver_id': f'driver_{i}',
                'current_node': node,
                'vehicle_type': random.choice(['car', 'motorcycle']),
                'capacity': random.randint(1, 3),
                'status': 'available'
            })
        return pd.DataFrame(drivers)

    def generate_orders(self, num_orders, time_window_size_minutes=60):
        orders = []
        for i in range(num_orders):
            origin_node = random.choice(list(self.graph.nodes()))
            destination_node = random.choice(list(self.graph.nodes()))
            while origin_node == destination_node:
                destination_node = random.choice(list(self.graph.nodes()))

            pickup_time = pd.Timestamp.now() + pd.Timedelta(minutes=random.randint(0, 120))
            delivery_latest = pickup_time + pd.Timedelta(minutes=random.randint(time_window_size_minutes, time_window_size_minutes * 3))

            orders.append({
                'order_id': f'order_{i}',
                'origin_node': origin_node,
                'destination_node': destination_node,
                'pickup_time_start': pickup_time,
                'pickup_time_end': pickup_time + pd.Timedelta(minutes=15),
                'delivery_time_latest': delivery_latest,
                'weight': random.uniform(0.1, 10.0),
                'volume': random.uniform(0.01, 0.5),
                'status': 'pending'
            })
        return pd.DataFrame(orders)

    def generate_traffic_data(self, num_records, duration_hours=24):
        traffic_data = []
        start_time = pd.Timestamp.now()
        edges = list(self.graph.edges(keys=True))
        for _ in range(num_records):
            u, v, key = random.choice(edges)
            travel_time = self.graph[u][v][key].get('travel_time', 1)
            traffic_factor = random.uniform(0.5, 2.0)
            traffic_data.append({
                'timestamp': start_time + pd.Timedelta(minutes=random.randint(0, duration_hours * 60)),
                'u': u,
                'v': v,
                'key': key,
                'current_travel_time': travel_time * traffic_factor
            })
        return pd.DataFrame(traffic_data)

    def save_scenario(self, drivers_df, orders_df, traffic_df, filename='tehran_fleet_scenario.pkl'):
        scenario_data = {
            'graph': self.graph,
            'drivers': drivers_df,
            'orders': orders_df,
            'traffic': traffic_df
        }
        with open(f'data_nexus/simulation_scenarios/{filename}', 'wb') as f:
            pickle.dump(scenario_data, f)
        print(f"Scenario saved to {filename}")

if __name__ == '__main__':
    generator = TehranFleetEnvironmentGenerator('conf/osm_processing_config.yaml')
    num_drivers = 50
    num_orders = 200
    num_traffic_records = 1000

    drivers = generator.generate_drivers(num_drivers)
    orders = generator.generate_orders(num_orders)
    traffic = generator.generate_traffic_data(num_traffic_records)

    generator.save_scenario(drivers, orders, traffic)
