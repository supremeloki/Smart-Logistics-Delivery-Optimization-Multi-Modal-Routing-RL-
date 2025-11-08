import requests
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphHopperAPIClient:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)["graphhopper"]
        self.base_url = self.config["host"] + "route"
        self.api_key = self.config["api_key"]

    def _load_config(self, config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get_route(
        self,
        origin,
        destination,
        vehicle_profile="car",
        instructions=True,
        calc_points=True,
        avoid_segments=None,
        preferred_speed=None,
    ):
        params = {
            "point": [f"{origin[0]},{origin[1]}", f"{destination[0]},{destination[1]}"],
            "vehicle": vehicle_profile,
            "instructions": instructions,
            "calc_points": calc_points,
            "locale": "en",
            "key": self.api_key,
            "ch.disable": True,  # Disable Contraction Hierarchies for more dynamic routing if needed
        }

        if avoid_segments:
            params["avoid"] = "|".join([f"{u},{v}" for u, v in avoid_segments])

        # Custom hints for dynamic costs, if GraphHopper supports it via parameters
        # This part might need custom GraphHopper setup or a paid plan with custom models
        # For simplicity, we assume 'preferred_speed' could be a custom parameter
        if preferred_speed:
            params["details"] = f"road_class:average_speed:{preferred_speed}"

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.get(
                self.base_url, params=params, headers=headers, timeout=10
            )
            response.raise_for_status()
            route_data = response.json()
            return self._parse_route_response(route_data)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling GraphHopper API: {e}")
            return None

    def _parse_route_response(self, route_data):
        if not route_data or "paths" not in route_data or not route_data["paths"]:
            return None

        path = route_data["paths"][0]
        parsed_route = {
            "distance": path.get("distance"),
            "time_ms": path.get("time"),
            "time_seconds": path.get("time") / 1000 if path.get("time") else None,
            "points": path.get("points", {}).get("coordinates"),
            "instructions": [
                {
                    "text": instr[0],
                    "distance": instr[1],
                    "time_ms": instr[2],
                    "point_index": instr[3],
                }
                for instr in path.get("instructions", [])
            ],
        }
        return parsed_route


if __name__ == "__main__":
    # Example usage (requires GraphHopper server running locally or accessible)
    # Ensure a local GraphHopper instance is running at http://localhost:8989/
    # docker run -d -p 8989:8989 -v "$PWD/graphhopper_data":/data graphhopper/graphhopper:latest "de.nordrhein-westfalen-latest.osm.pbf"
    client = GraphHopperAPIClient("conf/routing_engine_config.yaml")
    origin_coords = [35.7152, 51.4043]  # Example Tehran coords
    destination_coords = [35.7774, 51.4190]

    route = client.get_route(origin_coords, destination_coords)
    if route:
        logger.info(f"Route distance: {route['distance']:.2f} meters")
        logger.info(f"Route time: {route['time_seconds']:.2f} seconds")
        logger.info(
            f"Number of points: {len(route['points']) if route['points'] else 0}"
        )
    else:
        logger.warning("Failed to retrieve route.")
