import datetime
import json
import yaml
import os
import logging
import asyncio
from typing import Dict, Any, Tuple
import redis
import numpy as np


# Mock components
class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        # Mock demand data
        self.client.set(
            "zone_demand:ZONE_A",
            json.dumps({"demand_score": 0.8, "available_drivers": 2}),
        )
        self.client.set(
            "zone_demand:ZONE_B",
            json.dumps({"demand_score": 0.3, "available_drivers": 10}),
        )
        # Mock traffic data
        self.client.set(
            "traffic_congestion:ROUTE_AB",
            json.dumps({"congestion_level": "medium", "delay_factor": 1.3}),
        )
        # Mock weather data
        self.client.set(
            "weather_forecast:ZONE_A",
            json.dumps({"condition": "clear", "impact_factor": 1.0}),
        )
        self.client.set(
            "weather_forecast:ZONE_B",
            json.dumps({"condition": "heavy_rain", "impact_factor": 1.5}),
        )
        # Mock driver wellbeing (for bonus calculation)
        self.client.set(
            "driver_wellbeing:driver_A",
            json.dumps({"fatigue_score": 0.6, "stress_score": 0.7}),
        )
        self.client.set(
            "driver_wellbeing:driver_B",
            json.dumps({"fatigue_score": 0.1, "stress_score": 0.2}),
        )
        # Mock Order data
        self.client.set(
            "order:NEW_ORDER_001",
            json.dumps(
                {
                    "order_id": "NEW_ORDER_001",
                    "origin_zone": "ZONE_A",
                    "destination_zone": "ZONE_B",
                    "base_price": 20.0,
                    "urgency_multiplier": 1.2,
                    "status": "pending",
                    "creation_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        self.client.set(
            "order:NEW_ORDER_002",
            json.dumps(
                {
                    "order_id": "NEW_ORDER_002",
                    "origin_zone": "ZONE_B",
                    "destination_zone": "ZONE_C",
                    "base_price": 15.0,
                    "urgency_multiplier": 1.0,
                    "status": "pending",
                    "creation_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}

    def get_all_features_by_group(self, feature_group: str):
        keys = self.client.keys(f"{feature_group}:*")
        return (
            {k.split(":")[1]: json.loads(self.client.get(k)) for k in keys}
            if keys
            else {}
        )

    def set_feature(self, feature_group: str, key: str, value: Dict[str, Any]):
        self.client.set(f"{feature_group}:{key}", json.dumps(value))


class MockGISService:
    def __init__(self):
        pass

    def get_route_id(self, origin_zone, destination_zone):
        return f"ROUTE_{origin_zone}-{destination_zone}"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicPricingEngine:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.pricing_config = self.config["environments"][environment][
            "dynamic_pricing_engine"
        ]

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        self.gis_service = MockGISService()

        logger.info("DynamicPricingEngine initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    dynamic_pricing_engine:
      enabled: true
      pricing_interval_seconds: 5
      max_price_multiplier: 3.0
      min_price_multiplier: 0.8
      demand_sensitivity: 0.5 # How much demand affects price
      supply_sensitivity: 0.4 # How much driver supply affects price
      traffic_sensitivity: 0.3
      weather_sensitivity: 0.2
      urgency_sensitivity: 0.8
      driver_bonus_factor: 0.1 # Base bonus as % of price
      driver_fatigue_bonus_multiplier: 0.2
      driver_stress_bonus_multiplier: 0.1
      customer_loyalty_discount_factor: 0.05
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _calculate_dynamic_price(
        self, order_data: Dict[str, Any], customer_data: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the dynamic price for an order based on various real-time factors.
        Returns (dynamic_price, multipliers_applied).
        """
        base_price = order_data.get("base_price", 10.0)
        origin_zone = order_data.get("origin_zone")
        destination_zone = order_data.get("destination_zone")

        # Initialize with base
        dynamic_price = base_price
        multipliers_applied = {"base_price": base_price}

        # 1. Demand & Supply (at origin and destination)
        origin_demand = self.feature_store.get_feature("zone_demand", origin_zone) or {}
        dest_demand = (
            self.feature_store.get_feature("zone_demand", destination_zone) or {}
        )

        avg_demand_score = (
            origin_demand.get("demand_score", 0.5)
            + dest_demand.get("demand_score", 0.5)
        ) / 2
        avg_available_drivers = (
            origin_demand.get("available_drivers", 5)
            + dest_demand.get("available_drivers", 5)
        ) / 2

        demand_supply_factor = (
            avg_demand_score * self.pricing_config["demand_sensitivity"]
        ) - (
            (avg_available_drivers / 10.0) * self.pricing_config["supply_sensitivity"]
        )  # Assume 10 drivers is average
        price_multiplier_demand_supply = 1 + np.clip(
            demand_supply_factor, -0.5, 0.5
        )  # Limit impact
        dynamic_price *= price_multiplier_demand_supply
        multipliers_applied["demand_supply"] = price_multiplier_demand_supply

        # 2. Traffic Congestion
        route_id = self.gis_service.get_route_id(origin_zone, destination_zone)
        traffic_data = (
            self.feature_store.get_feature("traffic_congestion", route_id) or {}
        )
        traffic_delay_factor = traffic_data.get("delay_factor", 1.0)

        price_multiplier_traffic = (
            1 + (traffic_delay_factor - 1) * self.pricing_config["traffic_sensitivity"]
        )
        dynamic_price *= price_multiplier_traffic
        multipliers_applied["traffic"] = price_multiplier_traffic

        # 3. Weather Conditions
        origin_weather = (
            self.feature_store.get_feature("weather_forecast", origin_zone) or {}
        )
        dest_weather = (
            self.feature_store.get_feature("weather_forecast", destination_zone) or {}
        )

        avg_weather_impact_factor = (
            origin_weather.get("impact_factor", 1.0)
            + dest_weather.get("impact_factor", 1.0)
        ) / 2

        price_multiplier_weather = (
            1
            + (avg_weather_impact_factor - 1)
            * self.pricing_config["weather_sensitivity"]
        )
        dynamic_price *= price_multiplier_weather
        multipliers_applied["weather"] = price_multiplier_weather

        # 4. Urgency
        urgency_multiplier = order_data.get("urgency_multiplier", 1.0)
        dynamic_price *= (
            urgency_multiplier * self.pricing_config["urgency_sensitivity"]
        )  # Urgency directly increases price
        multipliers_applied["urgency"] = urgency_multiplier

        # 5. Customer Loyalty Discount (if customer_data provided)
        if customer_data and customer_data.get("loyalty_status", "none") == "gold":
            discount_factor = self.pricing_config["customer_loyalty_discount_factor"]
            dynamic_price *= 1 - discount_factor
            multipliers_applied["loyalty_discount"] = 1 - discount_factor

        # Apply overall limits
        overall_multiplier = dynamic_price / base_price
        if overall_multiplier > self.pricing_config["max_price_multiplier"]:
            dynamic_price = base_price * self.pricing_config["max_price_multiplier"]
        elif overall_multiplier < self.pricing_config["min_price_multiplier"]:
            dynamic_price = base_price * self.pricing_config["min_price_multiplier"]

        return round(dynamic_price, 2), multipliers_applied

    def _calculate_driver_incentive_bonus(
        self, order_price: float, driver_id: str = None
    ) -> float:
        """
        Calculates an additional bonus for the driver, factoring in wellbeing.
        """
        base_bonus = order_price * self.pricing_config["driver_bonus_factor"]

        if driver_id:
            wellbeing_data = (
                self.feature_store.get_feature("driver_wellbeing", driver_id) or {}
            )
            fatigue_score = wellbeing_data.get("fatigue_score", 0)
            stress_score = wellbeing_data.get("stress_score", 0)

            fatigue_bonus = (
                fatigue_score
                * self.pricing_config["driver_fatigue_bonus_multiplier"]
                * order_price
            )
            stress_bonus = (
                stress_score
                * self.pricing_config["driver_stress_bonus_multiplier"]
                * order_price
            )

            base_bonus += fatigue_bonus + stress_bonus

        return round(base_bonus, 2)

    async def run_pricing_loop(self):
        """Main loop for dynamically pricing new or pending orders."""
        if not self.pricing_config["enabled"]:
            logger.info("Dynamic pricing engine is disabled.")
            return

        while True:
            logger.info("Checking for pending orders to price...")
            pending_orders = self.feature_store.get_all_features_by_group("order")
            pending_orders = {
                oid: o
                for oid, o in pending_orders.items()
                if o.get("status") == "pending"
            }

            for order_id, order_data in pending_orders.items():
                dynamic_price, multipliers = await self._calculate_dynamic_price(
                    order_data
                )
                driver_bonus = await self._calculate_driver_incentive_bonus(
                    dynamic_price
                )  # Can be calculated for a hypothetical driver or as a general offer

                order_data["dynamic_price"] = dynamic_price
                order_data["driver_estimated_bonus"] = driver_bonus
                order_data["pricing_multipliers"] = multipliers
                order_data["pricing_updated_at"] = (
                    datetime.datetime.utcnow().isoformat() + "Z"
                )
                order_data["status"] = "priced"  # Move to next state

                self.feature_store.set_feature("order", order_id, order_data)
                logger.info(
                    f"Order {order_id} priced: ${dynamic_price:.2f} (Base: ${order_data['base_price']:.2f}, Bonus: ${driver_bonus:.2f}). Multipliers: {multipliers}"
                )

            await asyncio.sleep(self.pricing_config["pricing_interval_seconds"])


if __name__ == "__main__":
    import redis

    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    redis:
      host: localhost
      port: 6379
      db: 0
    dynamic_pricing_engine:
      enabled: true
      pricing_interval_seconds: 3 # Faster for demo
      max_price_multiplier: 3.0
      min_price_multiplier: 0.8
      demand_sensitivity: 0.5
      supply_sensitivity: 0.4
      traffic_sensitivity: 0.3
      weather_sensitivity: 0.2
      urgency_sensitivity: 0.8
      driver_bonus_factor: 0.1
      driver_fatigue_bonus_multiplier: 0.2
      driver_stress_bonus_multiplier: 0.1
      customer_loyalty_discount_factor: 0.05
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Populating dummy data for DynamicPricingEngine.")

        # Populate Redis with mock data
        r.set(
            "zone_demand:ZONE_A",
            json.dumps({"demand_score": 0.8, "available_drivers": 2}),
        )  # High demand, low supply
        r.set(
            "zone_demand:ZONE_B",
            json.dumps({"demand_score": 0.3, "available_drivers": 10}),
        )  # Low demand, high supply
        r.set(
            "zone_demand:ZONE_C",
            json.dumps({"demand_score": 0.5, "available_drivers": 5}),
        )

        r.set(
            "traffic_congestion:ROUTE_ZONE_A-ZONE_B",
            json.dumps({"congestion_level": "high", "delay_factor": 1.5}),
        )
        r.set(
            "traffic_congestion:ROUTE_ZONE_B-ZONE_C",
            json.dumps({"congestion_level": "low", "delay_factor": 1.0}),
        )

        r.set(
            "weather_forecast:ZONE_A",
            json.dumps({"condition": "heavy_rain", "impact_factor": 1.5}),
        )
        r.set(
            "weather_forecast:ZONE_B",
            json.dumps({"condition": "clear", "impact_factor": 1.0}),
        )
        r.set(
            "weather_forecast:ZONE_C",
            json.dumps({"condition": "light_snow", "impact_factor": 1.3}),
        )

        r.set(
            "driver_wellbeing:driver_A",
            json.dumps({"fatigue_score": 0.6, "stress_score": 0.7}),
        )
        r.set(
            "driver_wellbeing:driver_B",
            json.dumps({"fatigue_score": 0.1, "stress_score": 0.2}),
        )

        # Orders to be priced
        r.set(
            "order:ORD_HILO",
            json.dumps(
                {  # High demand/traffic, low supply
                    "order_id": "ORD_HILO",
                    "origin_zone": "ZONE_A",
                    "destination_zone": "ZONE_B",
                    "base_price": 25.0,
                    "urgency_multiplier": 1.5,
                    "status": "pending",
                    "creation_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "order:ORD_NORM",
            json.dumps(
                {  # Normal conditions
                    "order_id": "ORD_NORM",
                    "origin_zone": "ZONE_B",
                    "destination_zone": "ZONE_C",
                    "base_price": 20.0,
                    "urgency_multiplier": 1.0,
                    "status": "pending",
                    "creation_time": datetime.datetime.utcnow().isoformat() + "Z",
                }
            ),
        )
        r.set(
            "customer:Alice",
            json.dumps({"customer_id": "Alice", "loyalty_status": "gold"}),
        )  # For loyalty discount

    except redis.exceptions.ConnectionError:
        print("Redis not running. Pricing engine will start with empty data.")

    async def main_pricing():
        engine = DynamicPricingEngine(config_file)
        print("Starting DynamicPricingEngine for 20 seconds...")
        try:
            await asyncio.wait_for(engine.run_pricing_loop(), timeout=20)
        except asyncio.TimeoutError:
            print("\nDynamicPricingEngine demo timed out after 20 seconds.")
        except KeyboardInterrupt:
            print("\nDynamicPricingEngine demo stopped by user.")

    asyncio.run(main_pricing())
