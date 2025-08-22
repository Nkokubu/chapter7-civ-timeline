from fastapi import FastAPI, Query
from typing import List, Optional

api = FastAPI(title="CivTimeline API")

@api.get("/health")
def health():
    return {"ok": True}

CIVS = [
    {"id": 1, "name": "Roman Republic/Empire", "region": "Europe", "start_year": -509, "end_year": 476},
    {"id": 2, "name": "Han Dynasty", "region": "East Asia", "start_year": -206, "end_year": 220},
]

@api.get("/civs")
def list_civs(region: Optional[List[str]] = Query(None),
              start: int = -3000, end: int = 2000):
    data = [c for c in CIVS if c["start_year"] <= end and c["end_year"] >= start]
    if region:
        data = [c for c in data if c["region"] in region]
    return {"items": data, "count": len(data)}