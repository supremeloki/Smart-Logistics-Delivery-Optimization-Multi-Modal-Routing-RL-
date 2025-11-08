import json
import os
import datetime

class GrafanaDashboardGenerator:
    def __init__(self, dashboard_title="Logistics Optimization Metrics", uid="logistics_opt_metrics", output_dir="output/monitoring"):
        self.dashboard_title = dashboard_title
        self.uid = uid
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.dashboard_template = self._get_base_dashboard_template()

    def _get_base_dashboard_template(self):
        return {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": "-- Grafana --",
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "description": "Real-time performance and operational metrics for the Smart Logistics Optimization System.",
            "editable": True,
            "gnetId": None,
            "graphTooltip": 1,
            "id": None,
            "links": [],
            "panels": [],
            "schemaVersion": 30,
            "style": "dark",
            "tags": ["logistics", "optimization", "metrics", "production"],
            "templating": {
                "list": []
            },
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "timepicker": {},
            "timezone": "browser",
            "title": self.dashboard_title,
            "uid": self.uid,
            "version": 1
        }

    def _add_panel(self, panel_type, title, targets, span=12, y_axis_label="", format="short"):
        panel_id = len(self.dashboard_template['panels']) + 1
        new_panel = {
            "id": panel_id,
            "gridPos": {"h": 8, "w": span, "x": (panel_id - 1) % (12 // span) * span, "y": ((panel_id - 1) * span) // 12 * 8},
            "title": title,
            "type": panel_type,
            "targets": [],
            "options": {},
            "fieldConfig": {
                "defaults": {
                    "unit": format,
                    "min": 0,
                    "custom": {}
                },
                "overrides": []
            }
        }

        if panel_type == "graph":
            new_panel["xaxis"] = {"mode": "time", "show": True, "name": None, "values": []}
            new_panel["yaxes"] = [
                {"$$hashKey": "object:98", "format": format, "label": y_axis_label, "logBase": 1, "show": True},
                {"$$hashKey": "object:99", "format": format, "label": None, "logBase": 1, "show": True}
            ]

        for i, target_data in enumerate(targets):
            new_panel['targets'].append({
                "datasource": "Prometheus", # Assuming a Prometheus data source
                "expr": target_data['expr'],
                "legendFormat": target_data.get('legendFormat', f"series {i+1}"),
                "refId": chr(ord('A') + i),
                "hide": False
            })
        self.dashboard_template['panels'].append(new_panel)

    def generate_dashboard(self):
        # Delivery Completion Rate
        self._add_panel("stat", "Delivery Completion Rate", 
                        [{"expr": "sum(delivery_completion_total) / sum(order_placed_total)", "legendFormat": "Completion Rate"}], 
                        span=4, format="percentunit")
        
        # Average Delivery Time
        self._add_panel("stat", "Avg Delivery Time (minutes)", 
                        [{"expr": "rate(delivery_time_seconds_sum[5m]) / rate(delivery_time_seconds_count[5m]) / 60", "legendFormat": "Avg Time"}], 
                        span=4, format="m")

        # Route Efficiency
        self._add_panel("stat", "Route Efficiency (%)", 
                        [{"expr": "(1 - rate(deviation_distance_total[5m]) / rate(planned_distance_total[5m])) * 100", "legendFormat": "Efficiency"}], 
                        span=4, format="percent")

        # Orders Status Over Time (Graph)
        self._add_panel("graph", "Orders Status Over Time", [
            {"expr": "sum(logistics_orders_pending_count)", "legendFormat": "Pending"},
            {"expr": "sum(logistics_orders_in_transit_count)", "legendFormat": "In Transit"},
            {"expr": "sum(logistics_orders_delivered_count)", "legendFormat": "Delivered"}
        ], y_axis_label="Number of Orders")

        # Driver Load & Availability (Graph)
        self._add_panel("graph", "Driver Load & Availability", [
            {"expr": "avg(logistics_driver_current_load_kg)", "legendFormat": "Avg Load (kg)"},
            {"expr": "sum(logistics_driver_available_count)", "legendFormat": "Available Drivers"},
            {"expr": "sum(logistics_driver_on_route_count)", "legendFormat": "On Route Drivers"}
        ], y_axis_label="Count / Load")

        # API Latency (Graph)
        self._add_panel("graph", "API Latency (Routing/Optimization)", [
            {"expr": "rate(http_request_duration_seconds_sum{job=\"logistics-api\", handler=\"/route\"}[5m]) / rate(http_request_duration_seconds_count{job=\"logistics-api\", handler=\"/route\"}[5m])", "legendFormat": "Routing API Avg Latency"},
            {"expr": "rate(http_request_duration_seconds_sum{job=\"logistics-api\", handler=\"/optimize_fleet\"}[5m]) / rate(http_request_duration_seconds_count{job=\"logistics-api\", handler=\"/optimize_fleet\"}[5m])", "legendFormat": "Optimize Fleet API Avg Latency"}
        ], y_axis_label="Latency (seconds)", format="s")

        output_filename = os.path.join(self.output_dir, f"{self.uid}.json")
        with open(output_filename, 'w') as f:
            json.dump(self.dashboard_template, f, indent=2)
        return output_filename

if __name__ == '__main__':
    generator = GrafanaDashboardGenerator()
    dashboard_file = generator.generate_dashboard()
    print(f"Grafana dashboard JSON generated at: {dashboard_file}")
    print("\nTo import: In Grafana, go to Dashboards -> Import -> Upload JSON file.")
