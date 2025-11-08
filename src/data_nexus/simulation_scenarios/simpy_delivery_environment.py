"""
SimPy-based Delivery Environment for Logistics Simulation.
This module provides the core simulation environment for multi-modal routing and RL training.
"""

import pandas as pd
import simpy
import datetime
import logging
from typing import List, Dict, Optional, Any
import random

logger = logging.getLogger(__name__)

class Order:
    """Represents a delivery order in the simulation."""

    def __init__(self, order_id: str, origin_node: int, destination_node: int,
                 pickup_time_start: datetime.datetime, pickup_time_end: datetime.datetime,
                 delivery_time_latest: datetime.datetime, weight: float, volume: float):
        self.order_id = order_id
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.pickup_time_start = pickup_time_start
        self.pickup_time_end = pickup_time_end
        self.delivery_time_latest = delivery_time_latest
        self.weight = weight
        self.volume = volume

        # Simulation state
        self.picked_up_time: Optional[datetime.datetime] = None
        self.delivered_time: Optional[datetime.datetime] = None
        self.assigned_driver: Optional[str] = None

    def is_delivered(self) -> bool:
        return self.delivered_time is not None

    def is_picked_up(self) -> bool:
        return self.picked_up_time is not None


class Driver:
    """Represents a delivery driver/vehicle in the simulation."""

    def __init__(self, driver_id: str, start_node: int, vehicle_type: str = "car",
                 capacity: float = 100.0, speed_mps: float = 11.1):  # 40 km/h
        self.driver_id = driver_id
        self.start_node = start_node
        self.vehicle_type = vehicle_type
        self.capacity = capacity
        self.speed_mps = speed_mps

        # Simulation state
        self.current_node = start_node
        self.current_load = 0.0
        self.assigned_orders: List[str] = []
        self.is_available = True
        self.total_distance = 0.0

    def can_take_order(self, order: Order) -> bool:
        """Check if driver can take this order."""
        return (self.current_load + order.weight <= self.capacity and
                self.is_available and
                len(self.assigned_orders) < 5)  # Max 5 orders per driver


class SimpyDeliveryEnvironment:
    """Main simulation environment using SimPy for discrete event simulation."""

    def __init__(self, config_path: str, scenario_path: Optional[str] = None,
                 router: Optional[Any] = None, rl_agent_inferer: Optional[Any] = None):
        self.config_path = config_path
        self.scenario_path = scenario_path
        self.router = router
        self.rl_agent_inferer = rl_agent_inferer

        # Simulation components
        self.env = None
        self.drivers: Dict[str, Driver] = {}
        self.orders: Dict[str, Order] = {}
        self.metrics_history = []

        # Configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load simulation configuration."""
        # Default config
        return {
            "simulation": {
                "time_limit": 3600,  # 1 hour
                "step_size": 60,     # 1 minute steps
            },
            "environment": {
                "graph_path": "data_nexus/road_network_graph/preprocessed_tehran_graph.gml"
            }
        }

    def reset_simpy_env(self, initial_drivers: List[Driver], initial_orders: List[Order]):
        """Reset the simulation environment with new drivers and orders."""
        self.env = simpy.Environment()
        self.drivers = {d.driver_id: d for d in initial_drivers}
        self.orders = {o.order_id: o for o in initial_orders}
        self.metrics_history = []

        # Initialize driver processes
        for driver in self.drivers.values():
            self.env.process(self._driver_process(driver))

        logger.info(f"Simulation reset with {len(self.drivers)} drivers and {len(self.orders)} orders")

    def _driver_process(self, driver: Driver):
        """SimPy process for a driver."""
        while True:
            # Find available orders
            available_orders = [o for o in self.orders.values()
                              if not o.is_picked_up() and o.assigned_driver is None]

            if available_orders and driver.can_take_order(available_orders[0]):
                # Take an order (simple strategy - take first available)
                order = available_orders[0]
                order.assigned_driver = driver.driver_id
                driver.assigned_orders.append(order.order_id)
                driver.is_available = False

                # Simulate pickup
                pickup_time = random.uniform(5, 15)  # 5-15 minutes
                yield self.env.timeout(pickup_time)
                order.picked_up_time = datetime.datetime.now()

                # Simulate delivery
                delivery_time = random.uniform(10, 30)  # 10-30 minutes
                yield self.env.timeout(delivery_time)
                order.delivered_time = datetime.datetime.now()

                # Update driver state
                driver.current_load -= order.weight
                driver.assigned_orders.remove(order.order_id)
                driver.is_available = True
                driver.total_distance += random.uniform(5, 20)  # km

            else:
                # No orders available, wait
                yield self.env.timeout(5)  # Wait 5 minutes

    def run_simulation(self, until: float) -> pd.DataFrame:
        """Run the simulation until the specified time."""
        if self.env is None:
            logger.warning("Simulation environment not initialized")
            return pd.DataFrame()

        logger.info(f"Running simulation for {until} seconds")

        # Run simulation in steps to collect metrics
        step_size = self.config["simulation"]["step_size"]
        current_time = 0

        while current_time < until:
            # Run one step
            self.env.run(until=current_time + step_size)

            # Collect metrics
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)

            current_time += step_size

            if current_time >= until:
                break

        return pd.DataFrame(self.metrics_history)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current simulation metrics."""
        delivered_orders = sum(1 for o in self.orders.values() if o.is_delivered())
        pending_orders = sum(1 for o in self.orders.values() if not o.is_delivered())
        active_drivers = sum(1 for d in self.drivers.values() if not d.is_available)
        total_distance = sum(d.total_distance for d in self.drivers.values())

        return {
            "timestamp": self.env.now,
            "num_delivered_orders": delivered_orders,
            "num_pending_orders": pending_orders,
            "num_active_drivers": active_drivers,
            "num_total_drivers": len(self.drivers),
            "total_driver_distance": total_distance,
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state for RL agent."""
        return {
            "drivers": self.drivers,
            "orders": self.orders,
            "current_time": self.env.now if self.env else 0,
        }