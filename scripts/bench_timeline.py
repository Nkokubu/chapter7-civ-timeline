# scripts/bench_timeline.py
import time
from sqlmodel import select
from app.db.session import get_session
from app.models.event import Event
from streamlit_app import _build_timeline_bands

if __name__ == "__main__":
    with get_session() as s:
        evs = s.exec(select(Event).order_by(Event.year).limit(200)).all()
    t0 = time.perf_counter()
    fig = _build_timeline_bands(evs, (-1000, 1000))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"timeline_bands: {len(evs)} events -> {dt_ms:.1f} ms (fig={bool(fig)})")
