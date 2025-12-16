"""
station_camera_producer.py

Simulates the "Camera Subsystem + Station Data Manager":

 - Loads an image from a stop (camera).
 - Uses YOLOv8 to count people.
 - Uses Alejandro's models (age/gender/style) to obtain segments:
      * age_segment
      * gender_segment
      * product_segment
 - Sends an event to Kafka on the topic 'daedalus.station_events'.

Later on this script could:
 - be called periodically by a scheduler,
 - or become a service that watches a directory of images, etc.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaProducer
from ultralytics import YOLO

import time
from ad_inference import analyze_image_for_ads


# ============================================================
# 1. Configuration
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
KAFKA_TOPIC = "daedalus.station_events"

# Base path where you have the station (camera) images
BASE_IMAGES_DIR = Path("E:/uni/Proyecto_jf/camaras")  # change if needed

# List of stations we want to simulate.
# You can reuse images or use different ones.
STATION_CONFIG = [
    {"station_id": "4861", "image_filename": "img_nino.jpg"},
    {"station_id": "1027", "image_filename": "img_conteo_prueba_2.jpg"},
    # Add more if you want:
    # {"station_id": "1234", "image_filename": "cam_1234.jpg"},
]

# Sending mode
LOOP_FOREVER = True  # If True: infinite loop until Ctrl+C. If False: just one round.
SLEEP_BETWEEN_EVENTS_SEC = 10    # time between stations (seconds)
SLEEP_BETWEEN_ROUNDS_SEC = 60    # time between full rounds (seconds)


# ============================================================
# 2. YOLO model (people counting)
# ============================================================

print("Loading YOLOv8 model (person detector)...")
yolo_model = YOLO("yolov8n.pt")  # lightweight version, enough for testing


def count_people(image_path: Path) -> int:
    """
    Returns the number of people detected in the image using YOLOv8.
    """
    results = yolo_model(str(image_path), classes=[0])  # class 0 = 'person'
    detections = results[0].boxes
    # Count only detections of class 'person' (just in case)
    num_people = sum(1 for c in detections.cls if int(c) == 0)
    return int(num_people)


# ============================================================
# 3. Kafka Producer
# ============================================================

def get_kafka_producer() -> KafkaProducer:
    """
    Creates a Kafka producer to send JSON events.
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    return producer


def send_station_event(
    producer: KafkaProducer,
    station_id: str,
    image_path: Path,
) -> None:
    """
    - Counts people with YOLO.
    - Gets audience and ad segments using Alejandro's models.
    - Builds the event and sends it to Kafka.
    """
    # 1) People counting
    people_count = count_people(image_path)

    # 2) Audience prediction and segments (Alejandro's models)
    #    This is what you tested and it returns something like:
    #    AudiencePrediction(...) and AdSegments(...)
    audience, segments = analyze_image_for_ads(image_path)

    # 3) Build event
    event = {
        "station_id": station_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "people_count": people_count,
        # segments consumed by your ad_selector_consumer
        "age_segment": segments.age_segment,
        "gender_segment": segments.gender_segment,
        "product_segment": segments.product_segment,
        # extra optional info, in case you want to use it later
        "predicted_age_num": float(audience.age),
        "predicted_gender_raw": audience.gender,
        "predicted_style_raw": audience.style,
        "image_filename": image_path.name,
    }

    print(f"\n Sending event for station {station_id}:")
    print(json.dumps(event, indent=2, ensure_ascii=False))

    # 4) Send to Kafka
    producer.send(KAFKA_TOPIC, event)
    producer.flush()

    print(" Event sent to Kafka.")


# ============================================================
# 4. Main test / simulation
# ============================================================

def main():
    """
    Periodically sends events for multiple stations.
    Uses the configuration in STATION_CONFIG.
    """
    producer = get_kafka_producer()

    # Validate that the images exist
    for cfg in STATION_CONFIG:
        img_path = BASE_IMAGES_DIR / cfg["image_filename"]
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

    print("\n Starting camera simulation for stations:")
    for cfg in STATION_CONFIG:
        print(f"  - Station {cfg['station_id']} → {cfg['image_filename']}")
    print(f"\nSending one event every {SLEEP_BETWEEN_EVENTS_SEC} s per station.")
    if LOOP_FOREVER:
        print("Mode: infinite loop (CTRL+C to stop)")
    else:
        print("Mode: single round and exit.\n")

    ronda = 0
    try:
        while True:
            ronda += 1
            print(f"\n===== ROUND {ronda} =====")

            for cfg in STATION_CONFIG:
                station_id = cfg["station_id"]
                image_path = BASE_IMAGES_DIR / cfg["image_filename"]

                send_station_event(
                    producer=producer,
                    station_id=station_id,
                    image_path=image_path,
                )

                # Pause between stations
                time.sleep(SLEEP_BETWEEN_EVENTS_SEC)

            if not LOOP_FOREVER:
                print("\n Single round completed. Exiting.")
                break

            # Pause between full rounds
            print(f"\n Waiting {SLEEP_BETWEEN_ROUNDS_SEC} s before next round...")
            time.sleep(SLEEP_BETWEEN_ROUNDS_SEC)

    except KeyboardInterrupt:
        print("\n Simulation interrupted by user (CTRL+C).")
    finally:
        producer.flush()
        producer.close()
        print(" Kafka producer properly closed.")


if __name__ == "__main__":
    main()
