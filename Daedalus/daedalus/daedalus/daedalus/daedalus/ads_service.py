# daedalus/ads_service.py
import requests
from typing import List, Dict, Any

SUPABASE_URL = "https://rrbuwzcoiyncvmzctvnr.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJyYnV3emNvaXlu"
    "Y3ZtemN0dm5yIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzE1NzE1MywiZXhwIjoyMDc4NzMzMTUzfQ."
    "7tJ32D_TQeQ9xWVW8lYTv_T2c0N6U08adn2Llk-8iVk"
)

def _headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def list_ads() -> List[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/ads"
    params = {"select": "*", "order": "ad_id"}
    resp = requests.get(url, headers=_headers(), params=params)
    resp.raise_for_status()
    return resp.json()

def create_ad(
    ad_id: str,
    name: str,
    age_segment: str | None,
    gender_segment: str | None,
    product_segment: str | None,
    image_url: str | None,
    active: bool = True,
) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/ads"
    payload = [{
        "ad_id": ad_id,
        "name": name or None,
        "age_segment": age_segment or None,
        "gender_segment": gender_segment or None,
        "product_segment": product_segment or None,
        "image_url": image_url or None,
        "active": active,
    }]
    resp = requests.post(url, headers=_headers(), json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else {}

def update_ad_active(ad_id: str, active: bool) -> None:
    url = f"{SUPABASE_URL}/rest/v1/ads"
    params = {"ad_id": f"eq.{ad_id}"}
    payload = {"active": active}
    resp = requests.patch(url, headers=_headers(), params=params, json=payload)
    resp.raise_for_status()

def delete_ad(ad_id: str) -> None:
    url = f"{SUPABASE_URL}/rest/v1/ads"
    params = {"ad_id": f"eq.{ad_id}"}
    resp = requests.delete(url, headers=_headers(), params=params)
    resp.raise_for_status()
