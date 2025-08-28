# streamlit_app.py
from typing import List, Tuple, Dict
import streamlit as st
from sqlalchemy import and_, or_
from sqlalchemy.orm import selectinload
from sqlmodel import select

from app.db.session import get_session
from app.models.civilization import Civilization
from app.models.event import Event
import plotly.graph_objects as go

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

def _fmt_year(y: int) -> str:
    if y < 0:
        return f"{abs(y)} BCE"
    if y > 0:
        return f"{y} CE"
    return "0"

def _build_timeline_bands(events: List[Event], year_range: Tuple[int, int]):
    """Create a Plotly figure of horizontal bands (one per civ) within the selected range."""
    start_sel, end_sel = year_range

    # Collect min/max year per civ within the current selection
    by_civ: Dict[int, Dict] = {}
    for e in events:
        c = e.civilization
        if not c:
            continue
        bucket = by_civ.setdefault(c.id, {"name": c.name, "slug": c.slug, "region": c.region, "years": []})
        bucket["years"].append(e.year)

    rows = []
    for data in by_civ.values():
        # clip to selection
        start = max(min(data["years"]), start_sel)
        end   = min(max(data["years"]), end_sel)
        if start > end:
            continue  # nothing in range after clipping

        # avoid completely flat bars (Plotly doesn't render zero-length)
        dur = max(end - start, 0.1)
        rows.append({
            "label": data["name"],
            "base": start,
            "dur": dur,
            "start": start,
            "end": end,
        })

    if not rows:
        return None

    # Sort by earliest start, then name
    rows.sort(key=lambda r: (r["start"], r["label"]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        orientation="h",
        y=[r["label"] for r in rows],
        x=[r["dur"] for r in rows],      # bar length = duration
        base=[r["base"] for r in rows],  # where each bar starts on the x-axis
        customdata=[[ _fmt_year(r["start"]), _fmt_year(r["end"]), int(round(r["end"] - r["start"])) ] for r in rows],
        hovertemplate="<b>%{y}</b><br>Start: %{customdata[0]}<br>End: %{customdata[1]}<br>Duration: %{customdata[2]} years<extra></extra>",
        marker=dict(opacity=0.85),
    ))

    fig.update_layout(
        showlegend=False,
        barmode="overlay",
        margin=dict(l=140, r=20, t=10, b=40),
        height=350 + 12 * len(rows),  # grow height as rows increase
    )
    fig.update_xaxes(title="Year", range=[start_sel, end_sel], zeroline=True, zerolinewidth=1)
    fig.update_yaxes(title="", autorange="reversed")  # most recent on top feels nice; flip if you prefer

    return fig

def _build_civ_map(events: List[Event], by_civ: Dict[int, List[Event]]):
    """Return a Plotly geo scatter of civ centroids for the current filtered set."""
    civ_ids = list(by_civ.keys())
    if not civ_ids:
        return None

    # fetch centroids for the civs in view
    with get_session() as s:
        civs = s.exec(
            select(Civilization).where(Civilization.id.in_(civ_ids))
        ).all()

    lats, lons, names, hov = [], [], [], []
    for civ in civs:
        lat, lon = _civ_coords(civ)  # <-- use fallback-aware coords
        if lat is None or lon is None:
            continue
        evs = by_civ.get(civ.id, [])
        if not evs:
            continue
        years = [e.year for e in evs]
        y0, y1 = min(years), max(years)
        lats.append(lat)
        lons.append(lon)
        names.append(civ.name)
        hov.append(f"{civ.name}<br>Region: {civ.region}<br>In view: {y0} → {y1} ({y1 - y0} yrs)")

    if not lats:
        return None

    fig = go.Figure(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=names,
        hovertext=hov,
        hoverinfo="text",
        mode="markers",
        marker=dict(size=12, line=dict(width=1)),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        geo=dict(
            projection_type="natural earth",
            showcountries=True,
            showland=True,
            landcolor="rgb(229, 236, 246)",
            coastlinecolor="rgb(150,150,150)",
        ),
    )
    return fig

def _civ_coords(civ):
    # Prefer new columns
    lat = getattr(civ, "latitude", None)
    lon = getattr(civ, "longitude", None)
    # Fallback to legacy names if needed
    if lat is None or lon is None:
        lat = getattr(civ, "lat", None)
        lon = getattr(civ, "lon", None)
    return lat, lon

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

    # timeline bands (uses the same filtered events and year range)
    fig = _build_timeline_bands(events, year_range)
    if fig:
        st.subheader("Timeline bands")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events in the selected range to plot.")

    # fetch civs for the matching groups (ordered by name)
    civ_ids = list(by_civ.keys())
    if civ_ids:
        with get_session() as s:
            civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids)).order_by(Civilization.name)).all()
    else:
        civs = []

    st.caption(f"{len(civs)} civilization(s) match your filters.")

    # map (uses the same filtered events & civs)
    map_fig = _build_civ_map(events, by_civ)
    if map_fig:
        st.subheader("Map")
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.info("No mapped civilizations for the current filters.")


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

            # single-civ timeline band (optional)
            fig = _build_timeline_bands(rows, year_range)
            if fig:
                st.subheader("Timeline for this civilization")
                st.plotly_chart(fig, use_container_width=True)
            st.divider()

            st.write(f"{len(rows)} matching event(s) for {civ.name}:")
            for e in rows:
                with st.container(border=True):
                    st.write(f"**{e.year} — {e.title}**")
                    if e.summary:
                        st.write(e.summary)
                    if e.kind or e.tags:
                        st.caption(", ".join(filter(None, [e.kind, e.tags])))