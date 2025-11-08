import datetime
import math
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {config_path}: {e}")
        raise


def parse_iso_datetime(dt_string: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(dt_string)
    except ValueError:
        logger.error(f"Invalid ISO datetime string: {dt_string}")
        raise


def format_iso_datetime(dt_obj: datetime.datetime) -> str:
    return dt_obj.isoformat()


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_eta(
    current_location: tuple, destination_location: tuple, speed_mps: float
) -> float:
    if speed_mps <= 0:
        return float("inf")
    distance = haversine_distance(
        current_location[0],
        current_location[1],
        destination_location[0],
        destination_location[1],
    )
    return distance / speed_mps


def validate_coordinates(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


if __name__ == "__main__":
    logger.info("Testing common_helpers functions...")

    # Test load_config
    dummy_config_path = "conf/test_config.yaml"
    os.makedirs("conf", exist_ok=True)
    with open(dummy_config_path, "w") as f:
        f.write("setting1: value1\nsetting2: 123\n")

    try:
        config = load_config(dummy_config_path)
        logger.info(f"Loaded config: {config}")
        assert config["setting1"] == "value1"
        assert config["setting2"] == 123
    except Exception as e:
        logger.error(f"Error testing load_config: {e}")

    # Test datetime conversion
    now = datetime.datetime.now()
    iso_string = format_iso_datetime(now)
    parsed_dt = parse_iso_datetime(iso_string)
    logger.info(f"Original: {now}, ISO: {iso_string}, Parsed: {parsed_dt}")
    assert now.isoformat(timespec="seconds") == parsed_dt.isoformat(timespec="seconds")

    # Test haversine distance
    lat1, lon1 = 35.6892, 51.3890  # Tehran
    lat2, lon2 = 35.6881, 51.3917  # Nearby in Tehran
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    logger.info(
        f"Distance between ({lat1},{lon1}) and ({lat2},{lon2}): {dist:.2f} meters"
    )
    assert 200 < dist < 400

    # Test calculate_eta
    speed_kph = 30
    speed_mps = speed_kph / 3.6
    eta = calculate_eta((lat1, lon1), (lat2, lon2), speed_mps)
    logger.info(f"ETA for {dist:.2f}m at {speed_kph}km/h: {eta:.2f} seconds")
    assert 20 < eta < 60

    # Test validate_coordinates
    logger.info(f"Valid coords (90, 180): {validate_coordinates(90, 180)}")
    logger.info(f"Invalid coords (91, 181): {validate_coordinates(91, 181)}")
    assert validate_coordinates(90, 180)
    assert not validate_coordinates(91, 181)

    logger.info("All common_helpers tests passed.")
