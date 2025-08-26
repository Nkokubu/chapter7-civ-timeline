# streamlit_app.py
from typing import List, Tuple, Dict
import streamlit as st
from sqlalchemy import and_, or_
from sqlalchemy.orm import selectinload
from sqlmodel import select

from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event

st.set_page_config(page_title="Civilization Explorer", layout="wide")

# ---------- helpers ----------

def _distinct_regions() -> List[str]:
    with get_session() as s:
        rows = s.exec(select(Civilization.region).distinct().order_by(Civilization.region)).all()
    # .all() may return 1-tuples depending on driver
    return [r[0] if isinstance(r, tuple) else r for r in rows]

def _distinct_tags() -> List[str]:
    with get_session() as s:
        rows = s.exec(select(Event.tags).where(Event.tags.is_not(None))).all()
    seen, out = set(), []
    for r in rows:
        val = r[0] if isinstance(r, tuple) else r
        for t in (val or "").split(","):
            tag = t.strip()
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return sorted(out, key=str.lower)

def _split_csv(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out

def _query_events(year_range: Tuple[int, int], regions: List[str], tags: List[str]) -> List[Event]:
    start, end = year_range
    with get_session() as s:
        stmt = select(Event).options(selectinload(Event.civilization))
        conds = [Event.year >= start, Event.year <= end]

        if regions:
            # join to civ to filter by region (partial match so "Asia" matches "East Asia")
            stmt = stmt.join(Civilization)
            conds.append(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))

        if tags:
            conds.append(or_(*[Event.tags.ilike(f"%{t}%") for t in tags]))

        stmt = stmt.where(and_(*conds)).order_by(Event.year)
        return s.exec(stmt).all()

def _group_events_by_civ(events: List[Event]) -> Dict[int, List[Event]]:
    groups: Dict[int, List[Event]] = {}
    for e in events:
        if not e.civilization:
            continue
        groups.setdefault(e.civilization.id, []).append(e)
    return groups

# ---------- sidebar filters ----------

st.sidebar.header("Filters")
year_range = st.sidebar.slider("Year range", min_value=-3000, max_value=2000, value=(-500, 500), step=50)

# regions/tags are loaded once per run to keep the UI snappy
all_regions = _distinct_regions()
all_tags = _distinct_tags()

selected_regions = st.sidebar.multiselect("Regions", options=all_regions, default=[])
selected_tags = st.sidebar.multiselect("Tags", options=all_tags, default=[])

# ---------- routing state (list vs detail) ----------

if "selected_civ_id" not in st.session_state:
    st.session_state.selected_civ_id = None

def go_home():
    st.session_state.selected_civ_id = None

def go_detail(civ_id: int):
    st.session_state.selected_civ_id = civ_id

# ---------- main content ----------

if st.session_state.selected_civ_id is None:
    st.title("Civilizations")
    events = _query_events(year_range, selected_regions, _split_csv(selected_tags))
    by_civ = _group_events_by_civ(events)

    # fetch civs for the matching groups (ordered by name)
    civ_ids = list(by_civ.keys())
    if civ_ids:
        with get_session() as s:
            civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids)).order_by(Civilization.name)).all()
    else:
        civs = []

    st.caption(f"{len(civs)} civilization(s) match your filters.")

    # grid of cards (3 per row)
    cols_per_row = 3
    for i in range(0, len(civs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, civ in zip(cols, civs[i:i+cols_per_row]):
            with col:
                with st.container(border=True):
                    st.subheader(civ.name)
                    st.text(civ.region)
                    sample = by_civ.get(civ.id, [])[:3]
                    if sample:
                        st.write("Sample events:")
                        for e in sample:
                            st.write(f"- {e.year}: {e.title}")
                    st.button("View details", key=f"btn_{civ.id}", on_click=go_detail, args=(civ.id,))

else:
    # detail page for the selected civ
    with get_session() as s:
        civ = s.get(Civilization, st.session_state.selected_civ_id)
        if not civ:
            st.warning("Civilization not found.")
            go_home()
        else:
            st.button("← Back to list", on_click=go_home)
            st.title(civ.name)
            st.caption(civ.region)
            st.divider()

            # only this civ's events, still filtered by year/tags
            stmt = select(Event).where(Event.civilization_id == civ.id, Event.year >= year_range[0], Event.year <= year_range[1]).order_by(Event.year)
            if selected_tags:
                stmt = stmt.where(or_(*[Event.tags.ilike(f"%{t}%") for t in _split_csv(selected_tags)]))
            rows = s.exec(stmt).all()

            st.write(f"{len(rows)} matching event(s) for {civ.name}:")
            for e in rows:
                with st.container(border=True):
                    st.write(f"**{e.year} — {e.title}**")
                    if e.summary:
                        st.write(e.summary)
                    if e.kind or e.tags:
                        st.caption(", ".join(filter(None, [e.kind, e.tags])))