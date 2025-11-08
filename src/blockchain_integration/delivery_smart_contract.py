import yaml
import os
import logging
import random


# Mock Web3 library for demonstration purposes in the sandbox environment
class MockWeb3:
    def __init__(self, provider_url):
        self.provider_url = provider_url
        self.eth = MockEth()
        self.to_checksum_address = lambda x: x  # Simple pass-through for mock addresses
        logger.info(f"MockWeb3 initialized with provider: {provider_url}")

    def is_connected(self):
        return True  # Always connected in mock


class MockEth:
    def __init__(self):
        self.contract_instances = {}
        self.accounts = [f"0x{i:040x}" for i in range(5)]  # Dummy accounts
        self.default_account = self.accounts[0]

    def contract(self, address=None, abi=None):
        if address not in self.contract_instances:
            self.contract_instances[address] = MockContract(address, abi, self.accounts)
        return self.contract_instances[address]

    def get_transaction_receipt(self, tx_hash):
        # Simulate a transaction receipt
        return {
            "status": 1,  # Success
            "transactionHash": tx_hash,
            "blockNumber": random.randint(1000, 2000),
            "gasUsed": random.randint(20000, 50000),
        }


class MockContract:
    def __init__(self, address, abi, accounts):
        self.address = address
        self.abi = abi
        self.functions = MockContractFunctions()
        self.events = MockContractEvents()
        self.accounts = accounts
        self.storage = {  # Simulate smart contract storage
            "deliveries": {},  # delivery_id -> {status, driver, customer, payout, verified}
            "driver_balances": {acc: 0 for acc in accounts},
            "owner": accounts[0],
        }
        logger.info(f"MockContract initialized at {address}")

    def call(self, function_name, *args):
        # Simulate read-only contract calls
        if function_name == "getDeliveryStatus":
            delivery_id = args[0]
            return (
                self.storage["deliveries"]
                .get(delivery_id, {})
                .get("status", "NOT_FOUND")
            )
        elif function_name == "getDriverBalance":
            driver_address = args[0]
            return self.storage["driver_balances"].get(driver_address, 0)
        return "Mock Call Result"

    def transact(self, function_name, sender, *args):
        # Simulate state-changing transactions
        tx_hash = f"0x{random.getrandbits(256):064x}"
        logger.info(
            f"Mock Transaction '{function_name}' called by {sender} with args {args} -> {tx_hash}"
        )

        # Simulate state changes based on function_name
        if function_name == "createDelivery":
            delivery_id, driver_address, customer_address, payout_amount = args
            self.storage["deliveries"][delivery_id] = {
                "status": "CREATED",
                "driver": driver_address,
                "customer": customer_address,
                "payout": payout_amount,
                "verified": False,
            }
        elif function_name == "verifyDelivery":
            delivery_id = args[0]
            if delivery_id in self.storage["deliveries"]:
                self.storage["deliveries"][delivery_id]["verified"] = True
                self.storage["deliveries"][delivery_id]["status"] = "VERIFIED"
                driver_addr = self.storage["deliveries"][delivery_id]["driver"]
                payout = self.storage["deliveries"][delivery_id]["payout"]
                self.storage["driver_balances"][driver_addr] += payout

        return tx_hash  # Return dummy transaction hash


class MockContractFunctions:
    def __getattr__(self, name):
        # This makes it possible to call contract.functions.someFunction()
        def _mock_function_call(*args):
            return {
                "call": lambda: self.contract.call(name, *args),
                "transact": lambda tx_params: self.contract.transact(
                    name, tx_params["from"], *args
                ),
            }

        return _mock_function_call


class MockContractEvents:
    def __getattr__(self, name):
        # Mock for event filtering
        def _mock_event_filter(*args, **kwargs):
            return MockEventFilter(name)

        return _mock_event_filter


class MockEventFilter:
    def __init__(self, event_name):
        self.event_name = event_name

    def get_all_entries(self):
        # Simulate historical events
        if self.event_name == "DeliveryVerified":
            return [
                {
                    "args": {
                        "deliveryId": "DEL001",
                        "driverAddress": "0x0000...1",
                        "payout": 100,
                    },
                    "blockNumber": 100,
                },
                {
                    "args": {
                        "deliveryId": "DEL002",
                        "driverAddress": "0x0000...2",
                        "payout": 120,
                    },
                    "blockNumber": 105,
                },
            ]
        return []


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeliverySmartContractIntegrator:
    def __init__(self, config_path="conf/environments/prod.yaml", environment="dev"):
        self.config = self._load_config(config_path)
        self.blockchain_config = self.config["environments"][environment]["blockchain"]

        # Mock Web3.py for sandbox environment
        self.w3 = MockWeb3(self.blockchain_config["provider_url"])

        self.contract_address = self.blockchain_config["contract_address"]
        self.contract_abi = self.blockchain_config[
            "contract_abi"
        ]  # Typically loaded from a file

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.contract_address),
            abi=self.contract_abi,
        )
        self.owner_address = self.w3.eth.accounts[
            0
        ]  # The account deploying/managing the contract
        logger.info("DeliverySmartContractIntegrator initialized.")

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(
                    """
environments:
  dev:
    blockchain:
      enabled: true
      provider_url: http://localhost:8545
      contract_address: "0xMockContractAddress123456789012345678901234567890"
      contract_abi: |
        [
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"},{"internalType":"address","name":"driverAddress","type":"address"},{"internalType":"address","name":"customerAddress","type":"address"},{"internalType":"uint256","name":"payoutAmount","type":"uint256"}],"name":"createDelivery","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"}],"name":"verifyDelivery","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"}],"name":"getDeliveryStatus","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
          {"inputs":[{"internalType":"address","name":"driverAddress","type":"address"}],"name":"getDriverBalance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
          {"anonymous":false,"inputs":[{"indexed":true,"internalType":"string","name":"deliveryId","type":"string"},{"indexed":true,"internalType":"address","name":"driverAddress","type":"address"},{"indexed":false,"internalType":"uint256","name":"payout","type":"uint256"}],"name":"DeliveryVerified","type":"event"}
        ]
"""
                )
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def create_delivery_record(
        self,
        delivery_id: str,
        driver_address: str,
        customer_address: str,
        payout_amount: int,
    ):
        if not self.blockchain_config["enabled"]:
            logger.info(
                f"Blockchain integration disabled. Skipping create_delivery_record for {delivery_id}."
            )
            return None

        try:
            tx_hash = self.contract.transact(
                "createDelivery",
                self.owner_address,
                delivery_id,
                self.w3.to_checksum_address(driver_address),
                self.w3.to_checksum_address(customer_address),
                payout_amount,
            )
            # In a real scenario, you'd wait for transaction receipt
            # receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)  # Mock receipt
            logger.info(
                f"Delivery {delivery_id} created on blockchain. Tx Hash: {tx_hash}, Status: {receipt['status']}"
            )
            return tx_hash
        except Exception as e:
            logger.error(
                f"Failed to create delivery record {delivery_id} on blockchain: {e}"
            )
            return None

    def verify_delivery_completion(self, delivery_id: str):
        if not self.blockchain_config["enabled"]:
            logger.info(
                f"Blockchain integration disabled. Skipping verify_delivery_completion for {delivery_id}."
            )
            return None

        try:
            tx_hash = self.contract.transact(
                "verifyDelivery", self.owner_address, delivery_id
            )
            # receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)  # Mock receipt
            logger.info(
                f"Delivery {delivery_id} verification initiated on blockchain. Tx Hash: {tx_hash}, Status: {receipt['status']}"
            )
            return tx_hash
        except Exception as e:
            logger.error(
                f"Failed to verify delivery completion {delivery_id} on blockchain: {e}"
            )
            return None

    def get_delivery_status(self, delivery_id: str) -> str:
        if not self.blockchain_config["enabled"]:
            return "N/A (Blockchain Disabled)"

        try:
            status = self.contract.call("getDeliveryStatus", delivery_id)
            logger.debug(f"Status for delivery {delivery_id}: {status}")
            return status
        except Exception as e:
            logger.error(f"Failed to get delivery status for {delivery_id}: {e}")
            return "ERROR"

    def get_driver_balance(self, driver_address: str) -> int:
        if not self.blockchain_config["enabled"]:
            return 0
        try:
            balance = self.contract.call(
                "getDriverBalance", self.w3.to_checksum_address(driver_address)
            )
            logger.debug(f"Balance for driver {driver_address}: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Failed to get driver balance for {driver_address}: {e}")
            return -1

    def monitor_delivery_events(self):
        if not self.blockchain_config["enabled"]:
            logger.info("Blockchain integration disabled. Skipping event monitoring.")
            return []

        try:
            event_filter = self.contract.events.DeliveryVerified.create_filter(
                fromBlock="latest"
            )
            # For mock, we'll just return some dummy historical entries
            events = event_filter.get_all_entries()  # This would poll in a real client
            logger.info(f"Monitored {len(events)} DeliveryVerified events.")
            return events
        except Exception as e:
            logger.error(f"Error monitoring delivery events: {e}")
            return []


if __name__ == "__main__":
    # Ensure dummy config file for dev environment
    config_file = "conf/environments/dev.yaml"
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(
                """
environments:
  dev:
    blockchain:
      enabled: true
      provider_url: http://localhost:8545
      contract_address: "0xMockContractAddress123456789012345678901234567890"
      contract_abi: |
        [
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"},{"internalType":"address","name":"driverAddress","type":"address"},{"internalType":"address","name":"customerAddress","type":"address"},{"internalType":"uint256","name":"payoutAmount","type":"uint256"}],"name":"createDelivery","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"}],"name":"verifyDelivery","outputs":[],"stateMutability":"nonpayable","type":"function"},
          {"inputs":[{"internalType":"string","name":"deliveryId","type":"string"}],"name":"getDeliveryStatus","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
          {"inputs":[{"internalType":"address","name":"driverAddress","type":"address"}],"name":"getDriverBalance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
          {"anonymous":false,"inputs":[{"indexed":true,"internalType":"string","name":"deliveryId","type":"string"},{"indexed":true,"internalType":"address","name":"driverAddress","type":"address"},{"indexed":false,"internalType":"uint256","name":"payout","type":"uint256"}],"name":"DeliveryVerified","type":"event"}
        ]
"""
            )

    integrator = DeliverySmartContractIntegrator(config_file)

    # Get dummy accounts for testing
    driver_address_1 = integrator.w3.eth.accounts[1]
    customer_address_1 = integrator.w3.eth.accounts[2]

    # --- Step 1: Create a delivery record ---
    print("\n--- Creating a delivery record ---")
    delivery_id_1 = "DEL_001_A"
    payout_1 = 100
    tx_create = integrator.create_delivery_record(
        delivery_id_1, driver_address_1, customer_address_1, payout_1
    )
    if tx_create:
        print(f"Transaction to create delivery sent: {tx_create}")

    # Check status
    status_created = integrator.get_delivery_status(delivery_id_1)
    print(f"Status after creation: {status_created}")

    # --- Step 2: Verify delivery completion ---
    print("\n--- Verifying delivery completion ---")
    tx_verify = integrator.verify_delivery_completion(delivery_id_1)
    if tx_verify:
        print(f"Transaction to verify delivery sent: {tx_verify}")

    # Check status and driver balance after verification
    status_verified = integrator.get_delivery_status(delivery_id_1)
    print(f"Status after verification: {status_verified}")
    balance_driver_1 = integrator.get_driver_balance(driver_address_1)
    print(f"Driver {driver_address_1}'s balance: {balance_driver_1}")

    # --- Step 3: Monitor events ---
    print("\n--- Monitoring Delivery Verified events ---")
    events = integrator.monitor_delivery_events()
    for event in events:
        print(
            f"  Event: Delivery {event['args']['deliveryId']} verified, Payout {event['args']['payout']} to {event['args']['driverAddress']}"
        )

    print("\nDelivery Smart Contract Integrator demo complete.")
