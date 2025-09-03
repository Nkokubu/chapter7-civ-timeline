import requests

BASE = "http://127.0.0.1:8000"

def test_events_window():
    r = requests.get(f"{BASE}/events", params={"start": -500, "end": 500, "limit": 1000})
    r.raise_for_status()
    data = r.json()
    assert 1 <= len(data) <= 1000

def test_region_and_tags():
    r = requests.get(f"{BASE}/events", params={"region": "Asia,Europe", "tags": "war", "limit": 1000})
    r.raise_for_status()
    assert isinstance(r.json(), list)