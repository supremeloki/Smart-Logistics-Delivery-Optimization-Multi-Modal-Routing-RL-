import datetime
import json
import yaml
import os
import logging
import random
import asyncio
import redis  # <--- Added: Missing import from previous turn
from typing import Dict, Any, List


# Mock Web3 library for demonstration purposes in the sandbox environment
class MockWeb3:
    def __init__(self, provider_url):
        self.provider_url = provider_url
        self.eth = MockEth()
        self.to_checksum_address = lambda x: x
        logging.debug(f"MockWeb3 initialized with provider: {provider_url}")

    def is_connected(self):
        return True


class MockEth:
    def __init__(self):
        self.contract_instances = {}
        self.accounts = [f"0x{i:040x}" for i in range(5)]
        self.default_account = self.accounts[0]

    def contract(self, address=None, abi=None):
        if address not in self.contract_instances:
            self.contract_instances[address] = MockProductTraceabilityContract(
                address, abi, self.accounts
            )
        return self.contract_instances[address]

    def get_transaction_receipt(self, tx_hash):
        return {
            "status": 1,
            "transactionHash": tx_hash,
            "blockNumber": random.randint(1000, 2000),
            "gasUsed": random.randint(20000, 50000),
        }


class MockProductTraceabilityContract:
    def __init__(self, address, abi, accounts):
        self.address = address
        self.abi = abi
        self.functions = MockProductTraceabilityContractFunctions(self)
        self.events = MockContractEvents()
        self.accounts = accounts
        self.product_trace_log = {}  # product_id -> list of trace events
        self.product_ownership = {}  # product_id -> current_owner
        logging.debug(f"Mock Product Traceability Contract initialized at {address}")

    def call(self, function_name, *args):
        if function_name == "getTraceHistory":
            product_id = args[0]
            return self.product_trace_log.get(product_id, [])
        elif function_name == "getCurrentOwner":
            product_id = args[0]
            return self.product_ownership.get(product_id, "unknown")
        return "Mock Call Result"

    def transact(self, function_name, sender, *args):
        tx_hash = f"0x{random.getrandbits(256):064x}"
        logging.debug(
            f"Mock Transaction '{function_name}' called by {sender} with args {args} -> {tx_hash}"
        )

        if function_name == "logProductEvent":
            product_id, event_type, location, details_json, new_owner = args
            event_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                "location": location,
                "details": json.loads(details_json),
                "recorded_by": sender,
                "tx_hash": tx_hash,
                "new_owner": new_owner,
            }
            if product_id not in self.product_trace_log:
                self.product_trace_log[product_id] = []
            self.product_trace_log[product_id].append(event_entry)
            self.product_ownership[product_id] = new_owner
            logging.info(
                f"Logged product event for {product_id}: {event_type} at {location}. New owner: {new_owner}"
            )

        return tx_hash


class MockProductTraceabilityContractFunctions:
    def __init__(self, contract_object):
        self.contract_object = contract_object

    def __getattr__(self, name):
        def _mock_function_call(*args):
            return {
                "call": lambda: self.contract_object.call(name, *args),
                "transact": lambda tx_params: self.contract_object.transact(
                    name, tx_params["from"], *args
                ),
            }

        return _mock_function_call


class MockContractEvents:
    def __getattr__(self, name):
        def _mock_event_filter(*args, **kwargs):
            return MockEventFilter(name)

        return _mock_event_filter


class MockEventFilter:
    def __init__(self, event_name):
        self.event_name = event_name

    def get_all_entries(self):
        return []


class MockFeatureStoreClient:
    def __init__(self, config_path, environment):
        self.client = redis.StrictRedis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        self.client.set(
            "product_info:PROD_A_SN001",
            json.dumps(
                {"name": "Item A", "batch": "BAT_X", "manufacturer": "MFG_CORP"}
            ),
        )

    def get_feature(self, feature_group: str, key: str):
        data = self.client.get(f"{feature_group}:{key}")
        return json.loads(data) if data else {}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductTraceabilityLedger:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.blockchain_config = self.config["environments"][environment][
            "product_traceability_ledger"
        ]

        self.w3 = MockWeb3(self.blockchain_config["provider_url"])

        self.contract_address = self.blockchain_config["trace_contract_address"]
        self.contract_abi = self.blockchain_config["trace_contract_abi"]

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.contract_address),
            abi=self.contract_abi,
        )
        self.sender_address = self.w3.eth.accounts[
            0
        ]  # The account making the trace entries

        self.feature_store = MockFeatureStoreClient(config_path, environment)
        logger.info("ProductTraceabilityLedger initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    blockchain_audit: # Re-using existing blockchain section for mock config simplicity
      enabled: true
      provider_url: http://localhost:8545
      audit_contract_address: "0xMockAuditContractAddr_ABCDE"
      audit_contract_abi: |
        [] # Minimal ABI, actual ABI from ProductTraceabilityLedger
    product_traceability_ledger:
      enabled: true
      provider_url: http://localhost:8545
      trace_contract_address: "0xMockProductTraceContractAddr_XYZ"
      trace_contract_abi: |
        [
          {"inputs":[{"internalType":"string","name":"productId","type":"string"},{"internalType":"string","name":"eventType","type":"string"},{"internalType":"string","name":"location","type":"string"},{"internalType":"string","name":"detailsJson","type":"string"},{"internalType":"string","name":"newOwner","type":"string"}],"name":"logProductEvent","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"productId","type":"string"}],"name":"getTraceHistory","outputs":[{"components":[{"internalType":"string","name":"eventType","type":"string"},{"internalType":"string","name":"location","type":"string"},{"internalType":"string","name":"timestamp","type":"string"},{"internalType":"string","name":"details","type":"string"},{"internalType":"string","name":"recordedBy","type":"string"},{"internalType":"string","name":"newOwner","type":"string"}],"internalType":"struct ProductTrace.TraceEvent[]","name":"","type":"tuple[]"}],"stateMutability":"view","type":"function"},
          {"inputs":[{"internalType":"string","name":"productId","type":"string"}],"name":"getCurrentOwner","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}
        ]
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def log_product_event(
        self,
        product_id: str,
        event_type: str,
        location: str,
        details: Dict[str, Any],
        new_owner: str,
    ) -> Dict[str, Any]:
        """
        Logs a new event for a product on the blockchain ledger.
        """
        if not self.blockchain_config["enabled"]:
            logger.info(
                f"Blockchain traceability disabled. Not logging event for {product_id}."
            )
            return {
                "status": "DISABLED",
                "message": "Blockchain traceability is disabled.",
            }

        try:
            details_json = json.dumps(details)
            tx_hash = self.contract.functions.logProductEvent(
                product_id, event_type, location, details_json, new_owner
            ).transact({"from": self.sender_address})

            tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)

            if tx_receipt and tx_receipt["status"] == 1:
                logger.info(
                    f"Successfully logged event '{event_type}' for product {product_id}. Tx: {tx_hash}"
                )
                return {
                    "status": "SUCCESS",
                    "tx_hash": tx_hash,
                    "block_number": tx_receipt["blockNumber"],
                }
            else:
                logger.error(
                    f"Failed to log event '{event_type}' for product {product_id}. Tx: {tx_hash}"
                )
                return {
                    "status": "FAILED",
                    "tx_hash": tx_hash,
                    "message": "Transaction failed or reverted.",
                }
        except Exception as e:
            logger.error(
                f"Error logging event for product {product_id}: {e}", exc_info=True
            )
            return {"status": "ERROR", "message": str(e)}

    def get_trace_history(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the full traceability history for a given product ID.
        """
        if not self.blockchain_config["enabled"]:
            logger.info(
                f"Blockchain traceability disabled. Not retrieving history for {product_id}."
            )
            return []

        try:
            history = self.contract.functions.getTraceHistory(product_id).call()
            return history
        except Exception as e:
            logger.error(
                f"Error retrieving trace history for product {product_id}: {e}",
                exc_info=True,
            )
            return []

    def get_current_owner(self, product_id: str) -> str:
        """
        Retrieves the current owner of a given product ID.
        """
        if not self.blockchain_config["enabled"]:
            logger.info(
                f"Blockchain traceability disabled. Not retrieving owner for {product_id}."
            )
            return "Disabled"

        try:
            owner = self.contract.functions.getCurrentOwner(product_id).call()
            return owner
        except Exception as e:
            logger.error(
                f"Error retrieving current owner for product {product_id}: {e}",
                exc_info=True,
            )
            return "Error"

    async def run_listener_loop(self):
        """
        In a real system, this would listen for blockchain events and update local caches.
        For mock, we just demonstrate logging and retrieval.
        """
        logger.info("ProductTraceabilityLedger listener loop (mock) started.")
        # In a real system, this would involve w3.eth.filter, get_logs, etc.
        await asyncio.sleep(self.blockchain_config.get("listen_interval_seconds", 5))

    async def async_method_placeholder(self):
        """
        Placeholder for async methods.
        """


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
    blockchain_audit:
      enabled: true
      provider_url: http://localhost:8545
      audit_contract_address: "0xMockAuditContractAddr_ABCDE"
      audit_contract_abi: |
        []
    product_traceability_ledger:
      enabled: true
      provider_url: http://localhost:8545
      trace_contract_address: "0xMockProductTraceContractAddr_XYZ"
      trace_contract_abi: |
        [
          {"inputs":[{"internalType":"string","name":"productId","type":"string"},{"internalType":"string","name":"eventType","type":"string"},{"internalType":"string","name":"location","type":"string"},{"internalType":"string","name":"detailsJson","type":"string"},{"internalType":"string","name":"newOwner","type":"string"}],"name":"logProductEvent","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"productId","type":"string"}],"name":"getTraceHistory","outputs":[{"components":[{"internalType":"string","name":"eventType","type":"string"},{"internalType":"string","name":"location","type":"string"},{"internalType":"string","name":"timestamp","type":"string"},{"internalType":"string","name":"details","type":"string"},{"internalType":"string","name":"recordedBy","type":"string"},{"internalType":"string","name":"newOwner","type":"string"}],"internalType":"struct ProductTrace.TraceEvent[]","name":"","type":"tuple[]"}],"stateMutability":"view","type":"function"},
          {"inputs":[{"internalType":"string","name":"productId","type":"string"}],"name":"getCurrentOwner","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}
        ]
"""
            )

    try:
        r = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis. Initializing ProductTraceabilityLedger.")
    except redis.exceptions.ConnectionError:
        print(
            "Redis not running. ProductTraceabilityLedger will start without Redis functionality."
        )

    async def main_traceability():
        ledger = ProductTraceabilityLedger(config_file)

        product_id_a = "PROD_XYZ_SN789"
        product_id_b = "PROD_ABC_SN101"

        print(f"\n--- Logging events for {product_id_a} ---")
        event1 = ledger.log_product_event(
            product_id_a,
            "MANUFACTURED",
            "Factory_A",
            {"batch_no": "B001", "operator": "John Doe"},
            "Warehouse_A_Owner",
        )
        print(f"Event 1 result: {event1}")
        await asyncio.sleep(0.1)

        event2 = ledger.log_product_event(
            product_id_a,
            "SHIPPED",
            "Warehouse_A",
            {"carrier": "ExpressLogistics", "tracking_id": "EL98765"},
            "InTransit_Owner",
        )
        print(f"Event 2 result: {event2}")
        await asyncio.sleep(0.1)

        event3 = ledger.log_product_event(
            product_id_a,
            "RECEIVED",
            "DistributionCenter_X",
            {"inspection_pass": True, "condition": "good"},
            "DistributionCenter_Owner",
        )
        print(f"Event 3 result: {event3}")
        await asyncio.sleep(0.1)

        print(f"\n--- Logging events for {product_id_b} ---")
        event4 = ledger.log_product_event(
            product_id_b,
            "ASSEMBLED",
            "Factory_B",
            {"components_version": "v2.1"},
            "Factory_B_Owner",
        )
        print(f"Event 4 result: {event4}")
        await asyncio.sleep(0.1)

        print(f"\n--- Retrieving history for {product_id_a} ---")
        history_a = ledger.get_trace_history(product_id_a)
        for entry in history_a:
            print(
                f"- {entry['timestamp']} | {entry['event_type']} at {entry['location']} by {entry['recorded_by']} (Owner: {entry['new_owner']})"
            )
            print(f"  Details: {entry['details']}")

        print(
            f"\n--- Current owner of {product_id_a}: {ledger.get_current_owner(product_id_a)} ---"
        )
        print(
            f"--- Current owner of {product_id_b}: {ledger.get_current_owner(product_id_b)} ---"
        )

    asyncio.run(main_traceability())
