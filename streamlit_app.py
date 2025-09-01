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
import math
from typing import Optional


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

        # region filter (partial match so "Asia" hits "East Asia")
        if regions:
            stmt = stmt.join(Civilization)
            conds.append(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))

        # tags & kind filter (treat taxonomy words like "tech" as kind too)
        if tags:
            terms = [t.strip() for t in tags if t and t.strip()]
            kind_terms = [t.lower() for t in terms if t.lower() in KIND_TAXONOMY]
            tag_terms  = [t for t in terms if t.lower() not in KIND_TAXONOMY]

            preds = []
            if tag_terms:
                preds.append(or_(*[Event.tags.ilike(f"%{t}%") for t in tag_terms]))
            if kind_terms:
                preds.append(or_(*[Event.kind.ilike(k) for k in kind_terms]))  # case-insensitive

            if preds:
                # match if EITHER the tags contain any term OR the kind matches any taxonomy term selected
                conds.append(or_(*preds))

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

def _kind_counts(events: List[Event]) -> Dict[str, int]:
    counts = {k: 0 for k in KIND_TAXONOMY + ["other"]}
    for e in events:
        counts[_norm_kind(e.kind)] += 1
    return counts

def _build_compare_chart(name_a: str, cnt_a: Dict[str, int], name_b: str, cnt_b: Dict[str, int]) -> go.Figure:
    x = [name_a, name_b]
    fig = go.Figure()
    for kind in KIND_TAXONOMY + ["other"]:
        fig.add_trace(go.Bar(
            x=x,
            y=[cnt_a.get(kind, 0), cnt_b.get(kind, 0)],
            name=kind.title(),
            marker_color=KIND_COLORS.get(kind, KIND_COLORS["other"]),
        ))
    fig.update_layout(
        barmode="stack",
        margin=dict(l=20, r=20, t=10, b=10),
        height=420,
        legend_title_text="Event kind",
        yaxis_title="Events (count)",
    )
    return fig

# ---- hotspot helpers ----
def _century_start(y: int) -> int:
    # floor to the start of that century; works for BCE (negative) too
    return (math.floor(y / 100.0)) * 100

def _century_starts_in_range(year_range: Tuple[int, int]) -> List[int]:
    s, e = year_range
    s = _century_start(s)
    e = _century_start(e)
    if e < s:
        s, e = e, s
    return list(range(s, e + 1, 100))

def _bin_events_by_century(events: List[Event], year_range: Tuple[int, int]) -> Dict[str, Dict[int, int]]:
    """
    Return { region -> { century_start -> count } } for events inside the selected window.
    Also includes an 'All (visible)' aggregate across regions.
    """
    centuries = _century_starts_in_range(year_range)
    if not centuries:
        return {}

    def _blank_map():
        return {c: 0 for c in centuries}

    per_region: Dict[str, Dict[int, int]] = {}
    all_map = _blank_map()

    for e in events:
        c = e.civilization
        if not c:
            continue
        cs = _century_start(e.year)
        if cs < centuries[0] or cs > centuries[-1]:
            continue
        region = (c.region or "Unknown").strip()
        per_region.setdefault(region, _blank_map())
        per_region[region][cs] += 1
        all_map[cs] += 1

    if any(v > 0 for v in all_map.values()):
        per_region = {"All (visible)": all_map, **per_region}

    return per_region

def _quintile_threshold(counts: List[int]) -> int | None:
    """80th percentile-ish threshold; use only positive counts to avoid 'all zero' highlighting."""
    positives = [c for c in counts if c > 0]
    if not positives:
        return None
    positives.sort()
    # index for ~80th percentile
    q_idx = max(0, math.ceil(0.8 * len(positives)) - 1)
    return positives[q_idx]

def _build_region_sparkline(title: str, series: Dict[int, int]) -> go.Figure | None:
    """
    Tiny sparkline: x = century starts, y = counts; top quintile highlighted as dots.
    """
    if not series:
        return None
    xs = sorted(series.keys())
    ys = [series[x] for x in xs]
    thr = _quintile_threshold(ys)

    fig = go.Figure()

    # base line
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
        showlegend=False,
    ))

    # highlight points (top quintile)
    if thr is not None:
        hx = [x for x, y in zip(xs, ys) if y >= thr]
        hy = [y for y in ys if y >= thr]
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers",
            marker=dict(size=9, color="#F39C12", line=dict(width=0.5)),
            name="hotspot",
            hovertemplate=f"<b>{title}</b><br>%{{x}} → %{{y}} events<extra></extra>",
            showlegend=False,
        ))

    # minimalist “sparkline” styling
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=8, b=8),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, fixedrange=True
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, rangemode="tozero"
        ),
        title=dict(text=title, x=0.02, y=0.95, xanchor="left", yanchor="top", font=dict(size=12)),
    )
    return fig



# ---- kind taxonomy, colors, icons ----
KIND_TAXONOMY = ["war", "dynasty", "tech", "culture", "economy", "religion"]

KIND_COLORS = {
    "war": "#EF553B",       # red-ish
    "dynasty": "#636EFA",   # blue
    "tech": "#00CC96",      # green
    "culture": "#AB63FA",   # purple
    "economy": "#FFA15A",   # orange
    "religion": "#19D3F3",  # cyan
    "other": "#B6E880",     # soft green fallback
}

KIND_SYMBOLS = {
    "war": "triangle-up",
    "dynasty": "square",
    "tech": "diamond",
    "culture": "circle",
    "economy": "cross",
    "religion": "star",
    "other": "circle-open",
}

def _norm_kind(k: str | None) -> str:
    """Normalize arbitrary kind strings into our taxonomy; unknowns -> 'other'."""
    if not k:
        return "other"
    kk = k.strip().lower()
    return kk if kk in KIND_TAXONOMY else "other"


def _build_timeline_bands(events: List[Event], year_range: Tuple[int, int]):
    """
    Bars = civilization lifespan (start_year→end_year, or full event span if missing),
    clipped to the selected year range. Dots = filtered events.
    """
    start_sel, end_sel = year_range

    # civs visible under current filters (region/tags/year)
    civ_ids = sorted({e.civilization.id for e in events if e.civilization})
    if not civ_ids:
        return None

    # fetch civ records
    with get_session() as s:
        civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids))).all()

    rows = []
    for civ in civs:
        span = _lifespan_for_civ(civ)  # uses start/end if present; else min/max of ALL events for that civ
        if not span:
            continue
        s_clip = max(span[0], start_sel)
        e_clip = min(span[1], end_sel)
        if s_clip > e_clip:
            continue
        dur = e_clip - s_clip
        if dur <= 0:
            dur = 0.1
        rows.append({
            "id": civ.id,
            "label": civ.name,
            "base": s_clip,
            "dur": dur,
            "start": s_clip,
            "end": e_clip,
        })

    if not rows:
        return None

    rows.sort(key=lambda r: r["label"])
    label_order = [r["label"] for r in rows]
    y_for_id = {r["id"]: r["label"] for r in rows}
    allowed_ids = set(y_for_id.keys())

    import plotly.graph_objects as go
    fig = go.Figure()

    # bars
    fig.add_trace(go.Bar(
        orientation="h",
        y=label_order,
        x=[r["dur"] for r in rows],
        base=[r["base"] for r in rows],
        customdata=[[ _fmt_year(r["start"]), _fmt_year(r["end"]), int(round(r["end"] - r["start"])) ] for r in rows],
        hovertemplate="<b>%{y}</b><br>Start: %{customdata[0]}<br>End: %{customdata[1]}<br>Duration: %{customdata[2]} years<extra></extra>",
        marker=dict(opacity=0.85),
        showlegend=False,
    ))

    # dots (by kind), only for civs that have bars and events in-range
    points_by_kind: Dict[str, Dict[str, list]] = {k: {"x": [], "y": [], "text": []} for k in KIND_TAXONOMY + ["other"]}

    for e in events:
        c = e.civilization
        if not c or c.id not in allowed_ids:
            continue
        if not (start_sel <= e.year <= end_sel):
            continue
        kind = _norm_kind(e.kind)
        points_by_kind[kind]["x"].append(e.year)
        points_by_kind[kind]["y"].append(y_for_id[c.id])
        points_by_kind[kind]["text"].append(f"{e.title}<br>{_fmt_year(e.year)}")

    for kind in KIND_TAXONOMY + ["other"]:
        data = points_by_kind[kind]
        if not data["x"]:
            continue
        fig.add_trace(go.Scatter(
            mode="markers",
            x=data["x"],
            y=data["y"],
            name=kind.title(),
            text=data["text"],
            hoverinfo="text",
            marker=dict(
                size=9,
                color=KIND_COLORS.get(kind, KIND_COLORS["other"]),
                symbol=KIND_SYMBOLS.get(kind, KIND_SYMBOLS["other"]),
                line=dict(width=0.5),
                opacity=0.95,
            ),
        ))

    fig.update_layout(
        showlegend=True,
        legend_title_text="Event kind",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        barmode="overlay",
        margin=dict(l=160, r=20, t=10, b=40),
        height=380 + 12 * len(rows),
    )
    fig.update_xaxes(title="Year", range=[start_sel, end_sel], zeroline=True, zerolinewidth=1)
    fig.update_yaxes(title="", autorange="reversed")
    return fig

def _lifespan_for_civ(civ: Civilization) -> Optional[Tuple[int, int]]:
    """Return (start_year, end_year) for a civ; fall back to full event range if needed."""
    if civ.start_year is not None and civ.end_year is not None:
        return civ.start_year, civ.end_year
    # fallback: derive from *all* events for this civ
    with get_session() as s:
        ys = s.exec(select(Event.year).where(Event.civilization_id == civ.id)).all()
    if not ys:
        return None
    years = [y[0] if isinstance(y, tuple) else y for y in ys]
    return min(years), max(years)

def _clip_interval(a: Tuple[int, int], window: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Clip [a0,a1] to window [w0,w1]; return None if no intersection."""
    a0, a1 = a; w0, w1 = window
    s, e = max(a0, w0), min(a1, w1)
    return (s, e) if s <= e else None

def _overlap_years(a: Tuple[int, int], b: Tuple[int, int], window: Tuple[int, int]) -> int:
    """Overlap length of two intervals, after clipping both to window."""
    a_ = _clip_interval(a, window)
    b_ = _clip_interval(b, window)
    if not a_ or not b_:
        return 0
    s = max(a_[0], b_[0]); e = min(a_[1], b_[1])
    return max(0, e - s)

def _neighbors_by_overlap(target: Civilization, window: Tuple[int, int], limit: int = 12) -> List[Dict]:
    """Return neighbors sorted by overlap (desc). Each item: {id,name,slug,overlap,ov_start,ov_end}."""
    t_span = _lifespan_for_civ(target)
    if not t_span:
        return []
    t_clip = _clip_interval(t_span, window)
    if not t_clip:
        return []

    with get_session() as s:
        others = s.exec(select(Civilization).where(Civilization.id != target.id)).all()

    rows: List[Dict] = []
    for c in others:
        c_span = _lifespan_for_civ(c)
        if not c_span:
            continue
        ov = _overlap_years(t_span, c_span, window)
        if ov <= 0:
            continue
        # also record the clipped overlap interval for display
        tc = _clip_interval(t_span, window); cc = _clip_interval(c_span, window)
        if not tc or not cc:
            continue
        s_ = max(tc[0], cc[0]); e_ = min(tc[1], cc[1])
        rows.append({"id": c.id, "name": c.name, "slug": c.slug, "overlap": ov, "ov_start": s_, "ov_end": e_})

    rows.sort(key=lambda r: (-r["overlap"], r["name"]))
    return rows[:limit]

def _build_neighbor_graph(target: Civilization, neighbors: List[Dict]) -> Optional[go.Figure]:
    """Simple radial network: target at center; neighbors on a circle; edge width scales with overlap."""
    if not neighbors:
        return None

    # positions
    cx, cy = 0.0, 0.0
    r = 1.0
    n = len(neighbors)
    if n == 1:
        angles = [0.0]
    else:
        angles = [2 * math.pi * i / n for i in range(n)]

    pos = []
    for ang in angles:
        pos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))

    max_ov = max(nb["overlap"] for nb in neighbors) or 1

    fig = go.Figure()

    # edges
    for (x, y), nb in zip(pos, neighbors):
        w = 1.0 + 6.0 * (nb["overlap"] / max_ov)
        fig.add_trace(go.Scatter(
            x=[cx, x], y=[cy, y],
            mode="lines",
            line=dict(width=w, color="rgba(120,120,120,0.6)"),
            hoverinfo="text",
            hovertext=f"{target.name} ↔ {nb['name']}<br>Overlap: {nb['overlap']} yrs",
            showlegend=False,
        ))

    # neighbor nodes
    fig.add_trace(go.Scatter(
        x=[x for x, _ in pos],
        y=[y for _, y in pos],
        mode="markers+text",
        marker=dict(size=12),
        text=[nb["name"] for nb in neighbors],
        textposition="bottom center",
        name="Contemporaries",
        hoverinfo="text",
        hovertext=[f"{nb['name']}<br>{_fmt_year(nb['ov_start'])} → {_fmt_year(nb['ov_end'])}<br>{nb['overlap']} yrs"
                   for nb in neighbors],
    ))

    # center node (target civ)
    fig.add_trace(go.Scatter(
        x=[cx], y=[cy],
        mode="markers+text",
        marker=dict(size=18, color="#F4B400"),  # gold-ish
        text=[target.name],
        textposition="top center",
        name="Selected",
        hoverinfo="text",
        hovertext=[f"{target.name}"],
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
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
compare_mode = st.sidebar.checkbox("Compare two civilizations", value=False)


#----------- Compare View ------------

if compare_mode:
    st.title("Compare civilizations")

    # Use same filters as list view
    tags_list = _split_csv(selected_tags)
    events = _query_events(year_range, selected_regions, tags_list)
    by_civ = _group_events_by_civ(events)

    civ_ids = list(by_civ.keys())
    if not civ_ids:
        st.info("No civilizations match the current filters. Try widening your year range or clearing tags/regions.")
        st.stop()

    # Fetch civ objects for the options (ordered)
    with get_session() as s:
        civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids)).order_by(Civilization.name)).all()

    # Try to default to Rome vs Han if available
    def find_index(slug_sub: str) -> int | None:
        slug_sub = slug_sub.lower()
        for i, c in enumerate(civs):
            if c.slug and slug_sub in c.slug.lower():
                return i
        return None

    idx_rome = find_index("roman") or find_index("rome")
    idx_han  = find_index("han")

    # Fallback to first two options if Rome/Han not both present
    if idx_rome is None or idx_han is None or idx_rome == idx_han:
        idx_rome, idx_han = 0, 1 if len(civs) > 1 else (0, 0)

    colA, colB = st.columns(2)
    with colA:
        civ_a = st.selectbox("Civilization A", civs, index=idx_rome, format_func=lambda c: c.name, key="cmp_civ_a")
    with colB:
        civ_b = st.selectbox("Civilization B", civs, index=idx_han,  format_func=lambda c: c.name, key="cmp_civ_b")

    if civ_a.id == civ_b.id:
        st.warning("Pick two different civilizations to compare.")
        st.stop()

    # Events in current window/filters for each civ
    ev_a = by_civ.get(civ_a.id, [])
    ev_b = by_civ.get(civ_b.id, [])

    # Quick stats (within current window)
    def _summ(evts: List[Event]) -> Dict[str, str]:
        if not evts:
            return {"count": "0", "span": "—"}
        yrs = [e.year for e in evts]
        return {"count": str(len(evts)), "span": f"{_fmt_year(min(yrs))} → {_fmt_year(max(yrs))}"}

    sA, sB = _summ(ev_a), _summ(ev_b)

    with colA:
        st.subheader(civ_a.name)
        st.caption(civ_a.region or "")
        st.metric("Events in view", sA["count"])
        st.caption(f"Window span: {sA['span']}")

    with colB:
        st.subheader(civ_b.name)
        st.caption(civ_b.region or "")
        st.metric("Events in view", sB["count"])
        st.caption(f"Window span: {sB['span']}")

    st.divider()
    st.subheader("Event types (stacked)")

    cnt_a = _kind_counts(ev_a)
    cnt_b = _kind_counts(ev_b)
    chart = _build_compare_chart(civ_a.name, cnt_a, civ_b.name, cnt_b)
    st.plotly_chart(chart, use_container_width=True)

    # Stop here so the normal list/detail view doesn't render beneath
    st.stop()

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

    # ---- Hotspot detection (sparklines per region) ----
    st.subheader("Hotspots by century")
    st.caption("Top-quintile centuries are highlighted (based on counts within each series).")

    binned = _bin_events_by_century(events, year_range)
    if not binned:
        st.info("No events in this window to analyze.")
    else:
        # lay out sparklines in a neat grid
        regions = list(binned.keys())
        cols_per_row = 3
        for i in range(0, len(regions), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, region in zip(cols, regions[i:i+cols_per_row]):
                with col:
                    fig = _build_region_sparkline(region, binned[region])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

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
    
            # --- Neighbors in time (Day 9) ---
            st.subheader("Neighbors in time")
            st.caption(f"Overlaps computed within the current window: {_fmt_year(year_range[0])} to {_fmt_year(year_range[1])}.")
            nbrs = _neighbors_by_overlap(civ, year_range)

            if not nbrs:
                st.info("No overlapping civilizations in this window.")
            else:
                # ranked list
                for nb in nbrs:
                    st.write(f"- **{nb['name']}** — {nb['overlap']} yrs "
                             f"({_fmt_year(nb['ov_start'])} → {_fmt_year(nb['ov_end'])})")

                # small network graph
                net = _build_neighbor_graph(civ, nbrs)
                if net:
                    st.plotly_chart(net, use_container_width=True)

            st.divider()  