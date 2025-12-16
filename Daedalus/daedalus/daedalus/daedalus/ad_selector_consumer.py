"""
ad_selector_consumer.py

Service "Data Manager + Ad Selector".

- Listens to events on Kafka (topic: daedalus.station_events)
  sent by station_camera_producer.py
- Stores each event in the station_events table (Supabase).
- Fetches the ad catalog from Supabase (table ads).
- Chooses the ad that best matches the audience segments from
  that catalog.
- Stores the selection in station_selected_ads.

Later on, this could be extended to:
- make the scoring more sophisticated,
- or move part of the logic to SQL (views, station-level aggregations, etc.).
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from kafka import KafkaConsumer

# ============================================================
# 1. Kafka configuration
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
KAFKA_TOPIC = "daedalus.station_events"
KAFKA_GROUP_ID = "daedalus-ad-selector"


# ============================================================
# 2. Supabase configuration
# ============================================================

SUPABASE_URL = "https://rrbuwzcoiyncvmzctvnr.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyYnV3emNvaXlu"
    "Y3ZtemN0dm5yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzE1NzE1MywiZXhwIjoyMDc4NzMzMTUzfQ."
    "7tJ32D_TQeQ9xWVW8lYTv_T2c0N6U08adn2Llk-8iVk"
)


def supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


# ============================================================
# 3. Ad catalog (read from Supabase)
# ============================================================

def fetch_ads_from_supabase() -> List[Dict[str, Any]]:
    """
    Returns the list of ads from the 'ads' table in Supabase.

    We only request the fields we need:
      ad_id, name, age_segment, gender_segment, product_segment, active
    """
    url = f"{SUPABASE_URL}/rest/v1/ads"
    params = {
        "select": "ad_id,name,age_segment,gender_segment,product_segment,active",
    }

    try:
        resp = requests.get(url, headers=supabase_headers(), params=params, timeout=10)
        resp.raise_for_status()
        ads = resp.json()
        print(f"📡 Ad catalog loaded from Supabase: {len(ads)} ads.")
        return ads
    except Exception as e:
        print("⚠ Error fetching ads from Supabase:", e)
        # If it fails, return an empty list so the selector can still run
        return []


# ============================================================
# 4. Ad selection logic
# ============================================================

def score_ad(ad: Dict[str, Any], event: Dict[str, Any]) -> int:
    """
    Returns a score of how well an ad matches the event.

    Simple heuristic:
      +2 if age_segment matches
      +2 if product_segment matches
      +1 if gender is compatible (same, 'Neutral' or 'All genders')

    Also prints a breakdown of the score to help debugging / demos.
    """
    score = 0

    # --- Age ---
    age_points = 0
    if ad.get("age_segment") == event.get("age_segment"):
        age_points = 2
        score += 2

    # --- Product ---
    product_points = 0
    if ad.get("product_segment") == event.get("product_segment"):
        product_points = 2
        score += 2

    # --- Gender ---
    gender_points = 0
    ad_gender = ad.get("gender_segment")
    ev_gender = event.get("gender_segment")
    if ad_gender in ("Neutral", "All genders") or ad_gender == ev_gender:
        gender_points = 1
        score += 1

    # Ranking log
    print(
        f"   · {ad.get('ad_id')} – {ad.get('name', '')} "
        f"=> age={age_points}, product={product_points}, gender={gender_points} → total={score}"
    )

    return score


def select_best_ad(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Chooses the active ad with the highest score, reading the catalog from Supabase,
    and prints the score breakdown.
    """
    ads = fetch_ads_from_supabase()
    # Filter only active ads (just in case)
    candidates = [ad for ad in ads if ad.get("active", True)]

    if not candidates:
        print(" There are no active ads in Supabase.")
        return None

    print("    Evaluating candidate ads:")

    best_ad = None
    best_score = -1

    for ad in candidates:
        s = score_ad(ad, event)
        if s > best_score:
            best_score = s
            best_ad = ad

    if best_ad is None or best_score <= 0:
        print(" No ad gets a positive score for this event.")
        return None

    print(
        f"    Best ad according to scoring: "
        f"{best_ad.get('ad_id')} – {best_ad.get('name', '')} (score={best_score})"
    )

    return best_ad


# ============================================================
# 5. Supabase helper functions
# ============================================================

def insert_station_event(event: Dict[str, Any]) -> None:
    """
    Stores the event in the station_events table.
    """
    payload = [
        {
            "station_id": event.get("station_id"),
            # Let Postgres interpret the ISO8601 timestamp or use now() if missing
            "ts": event.get("timestamp"),
            "people_count": event.get("people_count", 0),
            "age_segment": event.get("age_segment"),
            "gender_segment": event.get("gender_segment"),
            "product_segment": event.get("product_segment"),
        }
    ]
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/station_events",
        headers=supabase_headers(),
        json=payload,
    )
    resp.raise_for_status()


def insert_station_selected_ad(event: Dict[str, Any], ad: Dict[str, Any]) -> None:
    """
    Stores in station_selected_ads the chosen ad for this event.
    """
    payload = [
        {
            "station_id": event.get("station_id"),
            "ts": event.get("timestamp"),
            "ad_id": ad["ad_id"],
            "people_count": event.get("people_count", 0),
            "age_segment": event.get("age_segment"),
            "gender_segment": event.get("gender_segment"),
            "product_segment": event.get("product_segment"),
        }
    ]
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/station_selected_ads",
        headers=supabase_headers(),
        json=payload,
    )
    resp.raise_for_status()


# ============================================================
# 6. Main Kafka consumer
# ============================================================

def main() -> None:
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="latest",  # or "earliest" if you want to read from the beginning
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    print(" ad_selector consumer listening on Kafka...")
    for msg in consumer:
        event = msg.value
        print("\n Event received from Kafka:")
        print(json.dumps(event, indent=2, ensure_ascii=False))

        # 1) Always store the raw event
        try:
            insert_station_event(event)
            print("   → Event stored into station_events.")
        except Exception as e:
            print("   ⚠ Error storing into station_events:", e)

        # 2) Select best ad
        ad = select_best_ad(event)
        if ad is None:
            print("   ⚠ No compatible ad for this event.")
            continue

        print(f"    Chosen ad: {ad['ad_id']} – {ad.get('name', '')}")

        # 3) Store selection
        try:
            insert_station_selected_ad(event, ad)
            print("   → Selection stored into station_selected_ads.")
        except Exception as e:
            print("   ⚠ Error storing ad selection:", e)


if __name__ == "__main__":
    main()
