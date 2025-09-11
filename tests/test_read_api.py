# tests/test_read_api.py

def test_events_window(client):
    r = client.get("/events", params={"start": -500, "end": 500, "limit": 1000})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)

def test_region_and_tags(client):
    r = client.get("/events", params={"region": "Asia,Europe", "tags": "war", "limit": 1000})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)

