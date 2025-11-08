import yaml
import logging
import osmnx as ox
import networkx as nx
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSMNxProcessor:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        ox.settings.use_cache = True
        ox.settings.log_console = True
        ox.settings.cache_folder = self.config["osm_data"]["cache_dir"]

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def download_and_process_graph(self):
        logger.info("Starting OSM data download and graph processing...")
        bbox = self.config["osm_data"]["bounding_box"]
        self.config["osmnx"]["graph_type"]
        network_type = self.config["osmnx"]["network_type"]

        G = ox.graph_from_bbox(
            north=bbox[2],
            south=bbox[0],
            east=bbox[3],
            west=bbox[1],
            network_type=network_type,
            retain_all=True,
        )

        logger.info(
            f"Graph downloaded with {len(G.nodes)} nodes and {len(G.edges)} edges."
        )

        # Project graph
        G_proj = ox.project_graph(G)
        logger.info("Graph projected to local UTM coordinates.")

        # Add travel time to edges
        G_proj = ox.add_edge_speeds(G_proj)
        G_proj = ox.add_edge_travel_times(G_proj)
        logger.info("Added edge speeds and travel times.")

        # Simplify graph
        if self.config["osmnx"].get("simplify_graph", True):
            G_proj = ox.consolidate_intersections(
                G_proj, rebuild_graph=True, tolerance=15
            )
            logger.info(
                f"Graph simplified. New node count: {len(G_proj.nodes)}, edge count: {len(G_proj.edges)}"
            )

        # Add custom attributes if needed (e.g., centrality)
        if "calculate_centrality_measures" in self.config["preprocessing_steps"]:
            G_proj = self._add_centrality_measures(G_proj)

        self._save_graph(G_proj, self.config["graph_serialization"]["output_path"])
        return G_proj

    def get_graph(self):
        """Load or create the graph"""
        graph_path = self.config["graph_serialization"]["output_path"]
        if os.path.exists(graph_path):
            return nx.read_gml(graph_path)
        else:
            return self.download_and_process_graph()

    def _add_centrality_measures(self, G):
        logger.info("Calculating centrality measures...")
        try:
            closeness_centrality = nx.closeness_centrality(G)
            nx.set_node_attributes(G, closeness_centrality, "closeness_centrality")
            # You can add other centralities like betweenness, degree etc.
            logger.info("Closeness centrality calculated and added.")
        except Exception as e:
            logger.warning(f"Failed to calculate centrality measures: {e}")
        return G

    def _save_graph(self, G, path):
        output_format = self.config["graph_serialization"]["output_format"]
        if output_format == "gml":
            nx.write_gml(G, path)
        elif output_format == "graphml":
            ox.save_graphml(G, filepath=path)
        else:
            raise ValueError(f"Unsupported graph serialization format: {output_format}")
        logger.info(f"Graph saved to {path} in {output_format} format.")


if __name__ == "__main__":
    processor = OSMNxProcessor("conf/osm_processing_config.yaml")
    graph = processor.download_and_process_graph()
