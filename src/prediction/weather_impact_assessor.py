import datetime
import yaml
import os
import logging
import random


# Mock API client for hyper-local weather forecasts
class MockWeatherApiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        logger.info("MockWeatherApiClient initialized.")

    def get_forecast(
        self, lat: float, lon: float, time: datetime.datetime, lookahead_hours: int = 6
    ) -> dict:
        # Simulate hyper-local weather forecast for a point in time/space
        # Weather conditions: clear, light_rain, heavy_rain, light_snow, heavy_snow, fog, icy

        forecast = {
            "timestamp_utc": time.isoformat() + "Z",
            "latitude": lat,
            "longitude": lon,
            "predictions": [],
        }

        current_temp = random.uniform(-5, 30)  # Celsius

        for i in range(lookahead_hours):
            future_time = time + datetime.timedelta(hours=i)
            condition = "clear"
            visibility = random.uniform(10, 20)  # km
            precipitation_intensity = 0.0
            wind_speed = random.uniform(5, 20)  # kph

            # Introduce some variability and potential bad weather
            if random.random() < 0.15:  # 15% chance of some precipitation/fog
                choice = random.choice(["rain", "snow", "fog", "icy"])
                if choice == "rain":
                    condition = "light_rain" if random.random() < 0.7 else "heavy_rain"
                    precipitation_intensity = random.uniform(0.1, 5.0)
                elif choice == "snow":
                    condition = "light_snow" if random.random() < 0.7 else "heavy_snow"
                    precipitation_intensity = random.uniform(0.1, 10.0)
                    current_temp = random.uniform(-10, 2)
                elif choice == "fog":
                    condition = "fog"
                    visibility = random.uniform(0.1, 5)
                elif choice == "icy":
                    condition = "icy"
                    current_temp = random.uniform(-10, 0)

            # Wind speed also can affect conditions
            if wind_speed > 30 and condition == "clear":
                condition = "windy"

            forecast["predictions"].append(
                {
                    "forecast_time_utc": future_time.isoformat() + "Z",
                    "condition": condition,
                    "temperature_celsius": round(
                        current_temp + random.uniform(-2, 2), 1
                    ),
                    "precipitation_mm_per_hr": round(precipitation_intensity, 1),
                    "visibility_km": round(visibility, 1),
                    "wind_speed_kph": round(wind_speed + random.uniform(-5, 5), 1),
                }
            )
        return forecast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherImpactAssessor:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.weather_config = self.config["environments"][environment][
            "weather_impact_assessor"
        ]

        self.weather_api_client = MockWeatherApiClient(self.weather_config["api_key"])

        # Predefined impact factors (heuristic for creativity)
        self.impact_factors = {
            "light_rain": {
                "travel_time_multiplier": 1.1,
                "speed_reduction_kph": 5,
                "safety_score_penalty": 0.1,
            },
            "heavy_rain": {
                "travel_time_multiplier": 1.3,
                "speed_reduction_kph": 15,
                "safety_score_penalty": 0.3,
            },
            "light_snow": {
                "travel_time_multiplier": 1.4,
                "speed_reduction_kph": 20,
                "safety_score_penalty": 0.4,
            },
            "heavy_snow": {
                "travel_time_multiplier": 2.0,
                "speed_reduction_kph": 30,
                "safety_score_penalty": 0.6,
            },
            "fog": {
                "travel_time_multiplier": 1.2,
                "speed_reduction_kph": 10,
                "safety_score_penalty": 0.2,
            },
            "icy": {
                "travel_time_multiplier": 2.5,
                "speed_reduction_kph": 40,
                "safety_score_penalty": 0.8,
            },
            "windy": {
                "travel_time_multiplier": 1.05,
                "speed_reduction_kph": 3,
                "safety_score_penalty": 0.05,
            },
            "clear": {
                "travel_time_multiplier": 1.0,
                "speed_reduction_kph": 0,
                "safety_score_penalty": 0.0,
            },
        }

        logger.info("WeatherImpactAssessor initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    weather_impact_assessor:
      enabled: true
      api_key: MOCK_WEATHER_API_KEY_123
      lookahead_hours: 6
      min_safety_score: 0.2 # Below this, route is considered unsafe
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def assess_impact_on_route_segment(
        self,
        lat: float,
        lon: float,
        current_speed_kph: float,
        segment_free_flow_time_sec: float,
        segment_length_m: float,
    ) -> dict:
        """
        Assesses the weather impact for a specific geographical point and estimates
        adjusted travel time and safety.
        """
        if not self.weather_config["enabled"]:
            logger.info(
                "Weather Impact Assessor is disabled. Returning default impacts."
            )
            return {
                "adjusted_travel_time_sec": segment_free_flow_time_sec,
                "safety_score": 1.0,
                "weather_condition": "clear",
                "recommendation": "Normal operation.",
            }

        forecast_data = self.weather_api_client.get_forecast(
            lat, lon, datetime.datetime.utcnow(), self.weather_config["lookahead_hours"]
        )

        # For simplicity, we'll use the immediate forecast (first hour) for the current segment
        # In a real system, you'd integrate over the segment's expected travel time
        if not forecast_data["predictions"]:
            return {
                "adjusted_travel_time_sec": segment_free_flow_time_sec,
                "safety_score": 1.0,
                "weather_condition": "clear",
                "recommendation": "No forecast available, assuming normal.",
            }

        current_forecast = forecast_data["predictions"][0]
        condition = current_forecast["condition"]
        current_forecast["temperature_celsius"]
        current_forecast["visibility_km"]
        current_forecast["wind_speed_kph"]

        impact = self.impact_factors.get(condition, self.impact_factors["clear"])

        adjusted_time_sec = (
            segment_free_flow_time_sec * impact["travel_time_multiplier"]
        )

        # Apply speed reduction and re-calculate time if necessary
        effective_speed_kph = max(
            10, current_speed_kph - impact["speed_reduction_kph"]
        )  # Minimum speed of 10 kph
        if segment_length_m > 0:
            adjusted_time_sec_by_speed = (segment_length_m / 1000) / (
                effective_speed_kph / 3600
            )
            adjusted_time_sec = max(
                adjusted_time_sec, adjusted_time_sec_by_speed
            )  # Take the higher time

        safety_score = 1.0 - impact["safety_score_penalty"]

        recommendation = "Normal operation."
        if condition != "clear":
            recommendation = (
                f"Caution due to {condition.replace('_', ' ')}. Expect slower travel."
            )
        if safety_score < self.weather_config["min_safety_score"]:
            recommendation = f"Critical weather alert: {condition.replace('_', ' ')}! Consider re-routing or pausing travel for safety."

        logger.debug(
            f"Assessed weather impact for ({lat:.2f}, {lon:.2f}): Condition={condition}, Adjusted Time={adjusted_time_sec:.2f}s, Safety={safety_score:.2f}"
        )

        return {
            "adjusted_travel_time_sec": adjusted_time_sec,
            "safety_score": safety_score,
            "weather_condition": condition,
            "recommendation": recommendation,
            "original_free_flow_time_sec": segment_free_flow_time_sec,
        }

    def get_route_weather_summary(self, path_segments: list) -> dict:
        """
        Provides a summary of weather impact along a multi-segment route.
        `path_segments` is a list of dictionaries, each with 'lat', 'lon', 'free_flow_time_sec', 'length_m'.
        """
        total_adjusted_time_sec = 0
        min_safety_score = 1.0
        dominant_weather = {}  # {condition: count}
        recommendations = []

        for segment in path_segments:
            result = self.assess_impact_on_route_segment(
                segment["lat"],
                segment["lon"],
                segment.get(
                    "current_speed_kph", 60
                ),  # Default to 60 kph if not provided
                segment["free_flow_time_sec"],
                segment["length_m"],
            )
            total_adjusted_time_sec += result["adjusted_travel_time_sec"]
            min_safety_score = min(min_safety_score, result["safety_score"])
            dominant_weather[result["weather_condition"]] = (
                dominant_weather.get(result["weather_condition"], 0) + 1
            )
            if (
                result["recommendation"] not in recommendations
                and "Normal operation" not in result["recommendation"]
            ):
                recommendations.append(result["recommendation"])

        overall_condition = (
            max(dominant_weather, key=dominant_weather.get)
            if dominant_weather
            else "clear"
        )

        summary = {
            "total_adjusted_travel_time_sec": total_adjusted_time_sec,
            "overall_safety_score": min_safety_score,
            "overall_dominant_weather": overall_condition,
            "critical_recommendations": (
                recommendations if recommendations else ["Route appears clear."]
            ),
            "raw_segment_impacts": [
                self.assess_impact_on_route_segment(
                    s["lat"],
                    s["lon"],
                    s.get("current_speed_kph", 60),
                    s["free_flow_time_sec"],
                    s["length_m"],
                )
                for s in path_segments
            ],
        }
        logger.info(
            f"Route weather summary: Dominant weather '{overall_condition}', Safety {min_safety_score:.2f}, Adjusted time {total_adjusted_time_sec:.2f}s."
        )
        return summary


if __name__ == "__main__":
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    weather_impact_assessor:
      enabled: true
      api_key: MOCK_WEATHER_API_KEY_123
      lookahead_hours: 6
      min_safety_score: 0.5 # Lower for easier trigger in demo
"""
            )

    assessor = WeatherImpactAssessor(config_file)

    # --- Test 1: Assess impact on a single route segment ---
    lat, lon = 35.7, 51.4  # Tehran coordinates
    current_speed = 60.0  # kph
    free_flow_time = 120.0  # seconds
    segment_length = 1000.0  # meters

    print("--- Assessing single segment weather impact (expecting variability) ---")
    for _ in range(3):  # Run a few times to see different weather conditions
        impact_result = assessor.assess_impact_on_route_segment(
            lat, lon, current_speed, free_flow_time, segment_length
        )
        print(
            f"  Condition: {impact_result['weather_condition']}, Adjusted Time: {impact_result['adjusted_travel_time_sec']:.2f}s, Safety: {impact_result['safety_score']:.2f}, Rec: {impact_result['recommendation']}"
        )

    # --- Test 2: Get route weather summary for a multi-segment route ---
    print("\n--- Getting multi-segment route weather summary ---")
    path_segments = [
        {
            "lat": 35.70,
            "lon": 51.40,
            "free_flow_time_sec": 60,
            "length_m": 500,
            "current_speed_kph": 50,
        },
        {
            "lat": 35.71,
            "lon": 51.41,
            "free_flow_time_sec": 90,
            "length_m": 700,
            "current_speed_kph": 50,
        },
        {
            "lat": 35.72,
            "lon": 51.42,
            "free_flow_time_sec": 45,
            "length_m": 300,
            "current_speed_kph": 50,
        },
    ]

    for _ in range(2):  # Run a few times for different scenarios
        route_summary = assessor.get_route_weather_summary(path_segments)
        print(f"\nRoute Summary:")
        print(
            f"  Overall Dominant Weather: {route_summary['overall_dominant_weather']}"
        )
        print(f"  Overall Safety Score: {route_summary['overall_safety_score']:.2f}")
        print(
            f"  Total Adjusted Travel Time: {route_summary['total_adjusted_travel_time_sec']:.2f}s"
        )
        print(
            f"  Critical Recommendations: {', '.join(route_summary['critical_recommendations'])}"
        )

    print("\nWeather Impact Assessor demo complete.")
