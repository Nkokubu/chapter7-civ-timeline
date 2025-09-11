# streamlit_app.py
from typing import List, Tuple, Dict, Optional
import os, json, math, csv, io, textwrap, random, zipfile
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

from sqlalchemy import and_, or_
from sqlalchemy.orm import selectinload
from sqlmodel import select, SQLModel

from app.db.session import get_session
# If your session module exposes the engine, import it (helps auto-create tables)
try:
    from app.db.session import engine  # type: ignore
except Exception:
    engine = None  # fallback-safe if not exported

from app.models.civilization import Civilization
from app.models.event import Event

# --- Simple auth (env-guarded) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_PASSWORD = os.getenv("CIV_APP_PASSWORD")  # set in .env or environment

# ---------- tiny utilities ----------
def _fmt_year(y: int) -> str:
    if y < 0: return f"{abs(y)} BCE"
    if y > 0: return f"{y} CE"
    return "0"

def _split_csv(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out

# ---------- cached helpers sometimes used elsewhere ----------
@st.cache_data
def get_all_regions():
    with get_session() as s:
        rows = s.exec(
            select(Civilization.region).distinct().order_by(Civilization.region)
        ).all()
    return [r[0] if isinstance(r, tuple) else r for r in rows if r]

@st.cache_data
def get_all_tags():
    with get_session() as s:
        rows = s.exec(select(Event.tags).where(Event.tags.is_not(None)).distinct()).all()
    # Flatten/comma-split unique tags:
    seen = set()
    for row in rows:
        tagstr = row[0] if isinstance(row, tuple) else row
        for t in (tagstr or "").split(","):
            t = t.strip()
            if t:
                seen.add(t)
    return sorted(seen)

# ---------- Export/Import ----------
EXPORT_FIELDS = ["civ_slug", "title", "year", "kind", "summary", "tags"]

def _events_to_rows(events: List[Event]) -> List[dict]:
    rows = []
    for e in events:
        civ_slug = e.civilization.slug if (getattr(e, "civilization", None) and e.civilization.slug) else ""
        rows.append({
            "civ_slug": civ_slug,
            "title": e.title or "",
            "year": int(e.year),
            "kind": (e.kind or ""),
            "summary": (e.summary or ""),
            "tags": (e.tags or "")
        })
    return rows

def _export_events_csv_bytes(events: List[Event]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in _events_to_rows(events):
        w.writerow(r)
    return buf.getvalue().encode("utf-8-sig")

def _export_events_json_bytes(events: List[Event]) -> bytes:
    return json.dumps(_events_to_rows(events), indent=2).encode("utf-8")

def _parse_events_csv_bytes(data: bytes) -> List[dict]:
    text = data.decode("utf-8-sig")
    rdr = csv.DictReader(io.StringIO(text))
    out = []
    for r in rdr:
        try:
            out.append({
                "civ_slug": (r.get("civ_slug") or "").strip().lower(),
                "title": (r.get("title") or "").strip(),
                "year": int(r.get("year")),
                "kind": (r.get("kind") or "").strip(),
                "summary": (r.get("summary") or "").strip(),
                "tags": (r.get("tags") or "").strip(),
            })
        except Exception:
            continue
    return out

def _parse_events_json_bytes(data: bytes) -> List[dict]:
    try:
        arr = json.loads(data.decode("utf-8"))
        out = []
        for r in arr:
            out.append({
                "civ_slug": (r.get("civ_slug") or "").strip().lower(),
                "title": (r.get("title") or "").strip(),
                "year": int(r.get("year")),
                "kind": (r.get("kind") or "").strip(),
                "summary": (r.get("summary") or "").strip(),
                "tags": (r.get("tags") or "").strip(),
            })
        return out
    except Exception:
        return []

def _insert_events_from_rows(rows: List[dict], dry_run: bool = True) -> tuple[int,int,int]:
    inserted = skipped = missing = 0
    if not rows:
        return (0, 0, 0)
    with get_session() as s:
        civs = s.exec(select(Civilization)).all()
        slug_map = { (c.slug or "").lower(): c for c in civs }
        for r in rows:
            civ = slug_map.get(r["civ_slug"])
            if not civ:
                missing += 1
                continue
            exists = s.exec(
                select(Event).where(
                    Event.civilization_id == civ.id,
                    Event.title == r["title"],
                    Event.year == r["year"],
                )
            ).first()
            if exists:
                skipped += 1
                continue
            if not dry_run:
                s.add(Event(
                    civilization_id=civ.id,
                    title=r["title"],
                    year=r["year"],
                    kind=(r.get("kind") or None),
                    summary=(r.get("summary") or None),
                    tags=(r.get("tags") or None),
                ))
                inserted += 1
        if not dry_run and inserted:
            s.commit()
    return (inserted, skipped, missing)

# ---------- Query/model helpers ----------
KIND_TAXONOMY = ["war", "dynasty", "tech", "culture", "economy", "religion"]
KIND_COLORS = {
    "war": "#EF553B", "dynasty": "#636EFA", "tech": "#00CC96",
    "culture": "#AB63FA", "economy": "#FFA15A", "religion": "#19D3F3", "other": "#B6E880",
}
KIND_SYMBOLS = {
    "war": "triangle-up", "dynasty": "square", "tech": "diamond",
    "culture": "circle", "economy": "cross", "religion": "star", "other": "circle-open",
}

def _norm_kind(k: Optional[str]) -> str:
    if not k: return "other"
    kk = k.strip().lower()
    return kk if kk in KIND_TAXONOMY else "other"

def _query_events(year_range: Tuple[int, int], regions: List[str], tags: List[str]) -> List[Event]:
    start, end = year_range
    with get_session() as s:
        stmt = select(Event).options(selectinload(Event.civilization))
        conds = [Event.year >= start, Event.year <= end]

        if regions:
            stmt = stmt.join(Civilization)
            conds.append(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))

        if tags:
            terms = [t.strip() for t in tags if t and t.strip()]
            kind_terms = [t.lower() for t in terms if t.lower() in KIND_TAXONOMY]
            tag_terms  = [t for t in terms if t.lower() not in KIND_TAXONOMY]
            preds = []
            if tag_terms:
                preds.append(or_(*[Event.tags.ilike(f"%{t}%") for t in tag_terms]))
            if kind_terms:
                preds.append(or_(*[Event.kind.ilike(k) for k in kind_terms]))
            if preds:
                conds.append(or_(*preds))

        stmt = stmt.where(and_(*conds)).order_by(Event.year)
        return s.exec(stmt).all()

def _group_events_by_civ(events: List[Event]) -> Dict[int, List[Event]]:
    groups: Dict[int, List[Event]] = {}
    for e in events:
        if e.civilization:
            groups.setdefault(e.civilization.id, []).append(e)
    return groups

def _kind_counts(events: List[Event]) -> Dict[str, int]:
    counts = {k: 0 for k in KIND_TAXONOMY + ["other"]}
    for e in events:
        counts[_norm_kind(e.kind)] += 1
    return counts

def _lifespan_for_civ(civ: Civilization) -> Optional[Tuple[int, int]]:
    if civ.start_year is not None and civ.end_year is not None:
        return civ.start_year, civ.end_year
    with get_session() as s:
        ys = s.exec(select(Event.year).where(Event.civilization_id == civ.id)).all()
    if not ys:
        return None
    years = [y[0] if isinstance(y, tuple) else y for y in ys]
    return min(years), max(years)

def _clip_interval(a: Tuple[int, int], window: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    a0, a1 = a; w0, w1 = window
    s, e = max(a0, w0), min(a1, w1)
    return (s, e) if s <= e else None

def _overlap_years(a: Tuple[int, int], b: Tuple[int, int], window: Tuple[int, int]) -> int:
    a_ = _clip_interval(a, window)
    b_ = _clip_interval(b, window)
    if not a_ or not b_: return 0
    s = max(a_[0], b_[0]); e = min(a_[1], b_[1])
    return max(0, e - s)

def _neighbors_by_overlap(target: Civilization, window: Tuple[int, int], limit: int = 12) -> List[Dict]:
    t_span = _lifespan_for_civ(target)
    if not t_span:
        return []
    if not _clip_interval(t_span, window):
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
        tc = _clip_interval(t_span, window); cc = _clip_interval(c_span, window)
        if not tc or not cc:
            continue
        s_ = max(tc[0], cc[0]); e_ = min(tc[1], cc[1])
        rows.append({"id": c.id, "name": c.name, "slug": c.slug, "overlap": ov, "ov_start": s_, "ov_end": e_})
    rows.sort(key=lambda r: (-r["overlap"], r["name"]))
    return rows[:limit]

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

def _build_timeline_bands(events: List[Event], year_range: Tuple[int, int]):
    start_sel, end_sel = year_range
    civ_ids = sorted({e.civilization.id for e in events if e.civilization})
    if not civ_ids:
        return None
    with get_session() as s:
        civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids))).all()
    rows = []
    for civ in civs:
        span = _lifespan_for_civ(civ)
        if not span:
            continue
        s_clip = max(span[0], start_sel)
        e_clip = min(span[1], end_sel)
        if s_clip > e_clip:
            continue
        dur = e_clip - s_clip
        if dur <= 0: dur = 0.1
        rows.append({"id": civ.id, "label": civ.name, "base": s_clip, "dur": dur, "start": s_clip, "end": e_clip})
    if not rows:
        return None
    rows.sort(key=lambda r: r["label"])
    label_order = [r["label"] for r in rows]
    y_for_id = {r["id"]: r["label"] for r in rows}
    allowed_ids = set(y_for_id.keys())

    fig = go.Figure()
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

    points_by_kind: Dict[str, Dict[str, list]] = {k: {"x": [], "y": [], "text": []} for k in KIND_TAXONOMY + ["other"]}
    for e in events:
        c = e.civilization
        if not c or c.id not in allowed_ids: continue
        if not (start_sel <= e.year <= end_sel): continue
        kind = _norm_kind(e.kind)
        points_by_kind[kind]["x"].append(e.year)
        points_by_kind[kind]["y"].append(y_for_id[c.id])
        points_by_kind[kind]["text"].append(f"{e.title}<br>{_fmt_year(e.year)}")

    for kind in KIND_TAXONOMY + ["other"]:
        data = points_by_kind[kind]
        if not data["x"]: continue
        fig.add_trace(go.Scatter(
            mode="markers",
            x=data["x"], y=data["y"], name=kind.title(), text=data["text"], hoverinfo="text",
            marker=dict(
                size=9, color=KIND_COLORS.get(kind, KIND_COLORS["other"]),
                symbol=KIND_SYMBOLS.get(kind, KIND_SYMBOLS["other"]), line=dict(width=0.5), opacity=0.95,
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

def _civ_coords(civ):
    lat = getattr(civ, "latitude", None); lon = getattr(civ, "longitude", None)
    if lat is None or lon is None:
        lat = getattr(civ, "lat", None); lon = getattr(civ, "lon", None)
    return lat, lon

def _build_civ_map(events: List[Event], by_civ: Dict[int, List[Event]]):
    civ_ids = list(by_civ.keys())
    if not civ_ids: return None
    with get_session() as s:
        civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids))).all()
    lats, lons, names, hov = [], [], [], []
    for civ in civs:
        lat, lon = _civ_coords(civ)
        if lat is None or lon is None: continue
        evs = by_civ.get(civ.id, [])
        if not evs: continue
        years = [e.year for e in evs]
        y0, y1 = min(years), max(years)
        lats.append(lat); lons.append(lon); names.append(civ.name)
        hov.append(f"{civ.name}<br>Region: {civ.region}<br>In view: {y0} → {y1} ({y1 - y0} yrs)")
    if not lats: return None
    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=names, hovertext=hov, hoverinfo="text",
        mode="markers", marker=dict(size=12, line=dict(width=1)),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        geo=dict(projection_type="natural earth", showcountries=True, showland=True,
                 landcolor="rgb(229, 236, 246)", coastlinecolor="rgb(150,150,150)"),
    )
    return fig

# ---------- Narrative + quiz ----------
KIND_WEIGHTS = {"war":3,"dynasty":2,"tech":2,"culture":1,"economy":1,"religion":2,"other":1}
TAG_WEIGHTS = {"found":2,"collapse":2,"conquest":2,"edict":1,"trade":1,"invention":2,"iron":2,
               "metallurgy":1,"naval":1,"library":1,"writing":1}

def _event_score(e: Event) -> int:
    k = _norm_kind(getattr(e, "kind", None))
    score = KIND_WEIGHTS.get(k, 1)
    tags = (e.tags or "").lower()
    title = (e.title or "").lower()
    text = f"{title} {tags}"
    for kw, w in TAG_WEIGHTS.items():
        if kw in text:
            score += w
    if e.summary and len(e.summary) >= 40:
        score += 1
    return score

def _pick_top_events(events: List[Event], per_civ_cap: int = 3, total_cap: int = 14) -> List[Event]:
    by_civ: Dict[int, List[Event]] = {}
    for e in events:
        if e.civilization:
            by_civ.setdefault(e.civilization.id, []).append(e)
    chosen: List[Event] = []
    for _, evs in by_civ.items():
        evs_sorted = sorted(evs, key=lambda x: (-_event_score(x), x.year))
        chosen.extend(evs_sorted[:per_civ_cap])
    chosen = sorted(chosen, key=lambda x: (-_event_score(x), x.year))[:total_cap]
    chosen.sort(key=lambda x: x.year)
    return chosen

def _make_narrative(selected_events: List[Event], year_range: Tuple[int, int], regions: List[str], tags: List[str], target_words: int = 250) -> str:
    if not selected_events:
        return "No events match the current filters—try widening the year range or clearing some filters."
    s, e = year_range
    win_label = f"{_fmt_year(s)} to {_fmt_year(e)}"
    region_label = ", ".join(regions) if regions else "all regions"
    tags_label = ", ".join(tags) if tags else "varied themes"
    intro = (f"This view covers {win_label}, focusing on {region_label} with {tags_label}. "
             f"Below is a quick narrative sketch assembled from key events (sequence, not causation). ")
    sentences: List[str] = []
    for ev in selected_events:
        civ = ev.civilization.name if ev.civilization else "Unknown"
        year_txt = _fmt_year(ev.year)
        title = ev.title or "(untitled event)"
        if ev.summary:
            sline = f"In {year_txt}, {civ}: {title} — {ev.summary}"
        else:
            sline = f"In {year_txt}, {civ}: {title}."
        sentences.append(sline)
    out = intro
    word_budget_low, word_budget_high = 200, 300
    target = max(word_budget_low, min(target_words, word_budget_high))
    words = out.split()
    for sline in sentences:
        if len(words) + len(sline.split()) > target + 60:
            break
        out += " " + sline
        words = out.split()
    return textwrap.fill(out, width=90)

# ---- Sequence hints ----
MAJOR_KINDS = {"war", "dynasty", "religion", "tech"}
MAJOR_TAG_WORDS = ["found","founding","unification","conquest","collapse","fall","decline","rise","annex","reform","edict"]

def _looks_major(e: Event) -> bool:
    if _norm_kind(e.kind) in MAJOR_KINDS:
        return True
    tags = (e.tags or "").lower()
    return any(w in tags for w in MAJOR_TAG_WORDS)

def _sequence_hints(target: Civilization, target_events: List[Event], window: Tuple[int,int], neighbors: Optional[List[Dict]] = None, delta: int = 50, max_per_neighbor: int = 3, max_total: int = 50) -> List[Dict]:
    if not target_events:
        return []
    t_years = [e.year for e in target_events]
    w0, w1 = window
    if neighbors is None:
        neighbors = _neighbors_by_overlap(target, window)
    neighbor_ids = [nb["id"] for nb in neighbors]
    if not neighbor_ids:
        return []
    with get_session() as s:
        evs = s.exec(
            select(Event).options(selectinload(Event.civilization)).where(
                Event.civilization_id.in_(neighbor_ids), Event.year >= w0, Event.year <= w1
            ).order_by(Event.year)
        ).all()
    by_neighbor: Dict[int, List[Dict]] = {}
    for ev in evs:
        if not _looks_major(ev): continue
        d = min(abs(ev.year - ty) for ty in t_years)
        if d <= delta and ev.civilization:
            row = {"neighbor_id": ev.civilization.id, "neighbor_name": ev.civilization.name,
                   "year": ev.year, "title": ev.title, "kind": _norm_kind(ev.kind), "distance": d}
            by_neighbor.setdefault(ev.civilization.id, []).append(row)
    out: List[Dict] = []
    for _, items in by_neighbor.items():
        items.sort(key=lambda r: (r["distance"], r["year"]))
        out.extend(items[:max_per_neighbor])
    out.sort(key=lambda r: (r["distance"], r["year"], r["neighbor_name"]))
    return out[:max_total]

# ---- Diffusion ----
DIFFUSION_THEMES = {
    "ironworking": ["ironworking","iron","metallurgy","steel","smelt","bloomery","forge"],
    "christianity": ["christianity","christian","christianization","church"],
    "trade networks": ["trade","silk-road","maritime","naval","sea route","merchant"]
}

def _first_adoptions(theme_key: str, year_range: Tuple[int, int], regions: List[str]) -> List[dict]:
    terms = [t.strip().lower() for t in DIFFUSION_THEMES.get(theme_key, [theme_key])]
    start, end = year_range
    with get_session() as s:
        stmt = select(Event).options(selectinload(Event.civilization))
        conds = [Event.year >= start, Event.year <= end]
        if regions:
            stmt = stmt.join(Civilization)
            conds.append(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))
        text_preds = []
        for t in terms:
            text_preds.append(Event.tags.ilike(f"%{t}%"))
            text_preds.append(Event.title.ilike(f"%{t}%"))
        conds.append(or_(*text_preds))
        rows = s.exec(stmt.where(and_(*conds)).order_by(Event.year)).all()
    per: Dict[int, dict] = {}
    for e in rows:
        c = e.civilization
        if not c: continue
        if c.id not in per or e.year < per[c.id]["year"]:
            lat, lon = _civ_coords(c)
            per[c.id] = {"civ": c, "year": e.year, "title": e.title, "lat": lat, "lon": lon}
    adoptions = list(per.values())
    adoptions.sort(key=lambda d: d["year"])
    return adoptions

def _build_diffusion_timeline(adoptions: List[dict]) -> Optional[go.Figure]:
    if len(adoptions) < 2: return None
    names = [a["civ"].name for a in adoptions]
    years = [a["year"] for a in adoptions]
    y_idx = list(range(len(names)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=y_idx, mode="lines+markers",
        text=[f"{n}<br>{_fmt_year(y)}" for n, y in zip(names, years)],
        hoverinfo="text", line=dict(width=2), marker=dict(size=10), showlegend=False
    ))
    fig.update_yaxes(tickvals=y_idx, ticktext=names, title="")
    fig.update_xaxes(title="Year")
    fig.update_layout(height=320, margin=dict(l=160, r=20, t=10, b=40))
    return fig

def _build_diffusion_map(adoptions: List[dict]) -> Optional[go.Figure]:
    line_traces = []
    for a, b in zip(adoptions, adoptions[1:]):
        if None in (a["lat"], a["lon"], b["lat"], b["lon"]): continue
        line_traces.append(go.Scattergeo(
            lat=[a["lat"], b["lat"]], lon=[a["lon"], b["lon"]],
            mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False
        ))
    lats, lons, texts = [], [], []
    for a in adoptions:
        if a["lat"] is None or a["lon"] is None: continue
        lats.append(a["lat"]); lons.append(a["lon"]); texts.append(f'{a["civ"].name}<br>{_fmt_year(a["year"])}')
    if not lats: return None
    fig = go.Figure()
    for tr in line_traces: fig.add_trace(tr)
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, text=texts, hoverinfo="text", mode="markers", marker=dict(size=10, line=dict(width=1))))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        geo=dict(projection_type="natural earth", showcountries=True, showland=True, landcolor="rgb(229,236,246)", coastlinecolor="rgb(150,150,150)"),
    )
    return fig

# ---------- Curriculum pack helpers ----------
def _events_to_csv_bytes(events: List[Event]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["civ_slug","civ_name","region","year","title","kind","tags","summary"])
    for e in events:
        c = e.civilization
        if not c: continue
        w.writerow([c.slug or "", c.name or "", c.region or "", int(e.year), e.title or "", (e.kind or ""), (e.tags or ""), (e.summary or "")])
    return buf.getvalue().encode("utf-8")

def _civs_to_csv_bytes(civs: List[Civilization]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id","slug","name","region","start_year","end_year","latitude","longitude"])
    for c in civs:
        lat, lon = _civ_coords(c)
        w.writerow([c.id, c.slug or "", c.name or "", c.region or "",
                    "" if c.start_year is None else int(c.start_year),
                    "" if c.end_year   is None else int(c.end_year),
                    "" if lat is None else lat, "" if lon is None else lon])
    return buf.getvalue().encode("utf-8")

def _fig_to_png_bytes(fig, width=1200, height=700, scale=2) -> bytes:
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception as e:
        raise RuntimeError("PNG export requires the 'kaleido' package. Install with: poetry add kaleido") from e

def _build_curriculum_pack_zip(lens_spec: dict, selected_civs: List[Civilization], year_range: Tuple[int,int], regions: List[str], tags: List[str], include_quiz: bool=True, include_images: bool=True) -> tuple[bytes, str]:
    ev_all = _query_events(year_range, regions, tags)
    by_civ = _group_events_by_civ(ev_all)
    sel_ids = {c.id for c in selected_civs} if selected_civs else set(by_civ.keys())
    ev_sel = [e for e in ev_all if e.civilization and e.civilization.id in sel_ids]
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "created_at": created,
        "lens": {"name": lens_spec.get("name"), "year_range": list(year_range), "regions": regions, "tags": tags},
        "selection": {"civs": [{"id": c.id, "slug": c.slug, "name": c.name} for c in selected_civs], "event_count": len(ev_sel)},
        "notes": "This pack was generated from the current filters in the Streamlit app.",
    }
    civs_csv = _civs_to_csv_bytes(selected_civs)
    events_csv = _events_to_csv_bytes(ev_sel)

    image_blobs: list[tuple[str, bytes]] = []
    if include_images:
        fig_tl = _build_timeline_bands(ev_sel, year_range)
        if fig_tl: image_blobs.append(("images/timeline.png", _fig_to_png_bytes(fig_tl)))
        if by_civ:
            fig_map = _build_civ_map(ev_sel, {cid: evs for cid, evs in by_civ.items() if cid in sel_ids})
            if fig_map: image_blobs.append(("images/map.png", _fig_to_png_bytes(fig_map)))
        if len(sel_ids) == 2:
            id_a, id_b = list(sel_ids)
            ev_a = [e for e in ev_sel if e.civilization and e.civilization.id == id_a]
            ev_b = [e for e in ev_sel if e.civilization and e.civilization.id == id_b]
            name_a = next((c.name for c in selected_civs if c.id == id_a), "A")
            name_b = next((c.name for c in selected_civs if c.id == id_b), "B")
            chart = _build_compare_chart(name_a, _kind_counts(ev_a), name_b, _kind_counts(ev_b))
            image_blobs.append(("images/compare.png", _fig_to_png_bytes(chart, height=520)))

    quiz_blob: Optional[bytes] = None
    qz = st.session_state.get("quiz_state")
    if include_quiz and qz and qz.get("questions"):
        quiz_blob = json.dumps(qz, indent=2).encode("utf-8")

    readme = f"""Curriculum Pack
Generated: {created}

Lens:
  - Name    : {lens_spec.get('name') or '(none)'}
  - Years   : {year_range[0]} → {year_range[1]}
  - Regions : {', '.join(regions) if regions else 'All'}
  - Tags    : {', '.join(tags) if tags else 'None'}

Selected civilizations ({len(selected_civs)}):
{chr(10).join(['  - ' + (c.name or '(unnamed)') for c in selected_civs]) or '  (none)'}

Files:
  - meta.json          : metadata for this pack
  - civs.csv           : selected civilizations
  - events.csv         : filtered events for the selected civilizations
  - quiz.json          : (optional) quiz as shown in the app
  - images/timeline.png: (optional) timeline snapshot
  - images/map.png     : (optional) map snapshot
  - images/compare.png : (optional) if exactly two civs were selected

Notes:
  • Years use negative values for BCE (e.g., -500 = 500 BCE).
  • PNG export requires the 'kaleido' package in your environment.
"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"curriculum_pack_{stamp}.zip"
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme)
        z.writestr("meta.json", json.dumps(meta, indent=2))
        z.writestr("civs.csv", civs_csv)
        z.writestr("events.csv", events_csv)
        if quiz_blob: z.writestr("quiz.json", quiz_blob)
        for path, blob in image_blobs:
            z.writestr(path, blob)
    bio.seek(0)
    return bio.getvalue(), fname

# ---------- Quiz dataclasses ----------
@dataclass
class QuizEvent:
    id: int
    title: str
    civ: str
    year: int

@dataclass
class QuizCiv:
    id: int
    name: str
    year: int

def _rep_year_for_civ(civ: Civilization) -> Optional[int]:
    if civ.start_year is not None:
        return civ.start_year
    with get_session() as s:
        ys = s.exec(select(Event.year).where(Event.civilization_id == civ.id)).all()
    years = [y[0] if isinstance(y, tuple) else y for y in ys]
    return min(years) if years else None

def _load_quiz_pool(year_range, regions, tags) -> tuple[list[QuizEvent], list[QuizCiv]]:
    events = _query_events(year_range, regions, tags)
    by_civ = _group_events_by_civ(events)
    civ_ids = list(by_civ.keys())
    ev_pool: list[QuizEvent] = []
    seen_pairs = set()
    for e in events:
        if not e.civilization or e.year is None: continue
        key = (e.civilization.id, e.title, e.year)
        if key in seen_pairs: continue
        seen_pairs.add(key)
        ev_pool.append(QuizEvent(id=e.id, title=e.title, civ=e.civilization.name, year=e.year))
    cv_pool: list[QuizCiv] = []
    if civ_ids:
        with get_session() as s:
            civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids))).all()
        for c in civs:
            ry = _rep_year_for_civ(c)
            if ry is not None:
                cv_pool.append(QuizCiv(id=c.id, name=c.name, year=ry))
    return ev_pool, cv_pool

def _generate_quiz(seed: int, ev_pool: list[QuizEvent], cv_pool: list[QuizCiv]) -> dict:
    rnd = random.Random(seed)
    ev_candidates = [e for e in ev_pool]; rnd.shuffle(ev_candidates)
    which_first_qs = []; used_idx = set()
    for i in range(len(ev_candidates)):
        if len(which_first_qs) >= 5: break
        if i in used_idx: continue
        a = ev_candidates[i]; b = None
        for j in range(i+1, len(ev_candidates)):
            if j in used_idx: continue
            cand = ev_candidates[j]
            if cand.year != a.year:
                b = cand; used_idx.add(j); break
        if not b: continue
        used_idx.add(i)
        answer = 0 if a.year < b.year else 1
        which_first_qs.append({"kind":"which_first","a":{"title":a.title,"civ":a.civ,"year":a.year},"b":{"title":b.title,"civ":b.civ,"year":b.year},"answer":answer})
    civ_candidates = [c for c in cv_pool]; rnd.shuffle(civ_candidates)
    civ_year_qs = []
    all_years = sorted({c.year for c in cv_pool})
    for c in civ_candidates:
        if len(civ_year_qs) >= 5: break
        correct = c.year
        distractors = [y for y in all_years if y != correct]; rnd.shuffle(distractors)
        d = distractors[:3] if len(distractors) >= 3 else distractors
        choices = [correct] + d; rnd.shuffle(choices)
        answer = choices.index(correct)
        civ_year_qs.append({"kind":"civ_year","civ":{"name":c.name},"choices":choices,"answer":answer})
    questions = which_first_qs[:5] + civ_year_qs[:5]
    return {"seed":seed, "questions":questions, "responses":{}, "submitted":False, "score":None}

def _score_quiz(qz: dict) -> tuple[int,int]:
    score = 0; total = len(qz["questions"])
    for i, q in enumerate(qz["questions"]):
        ans = q["answer"]; sel = qz["responses"].get(i, None)
        if sel is not None and sel == ans: score += 1
    return score, total

# ---------- Lenses ----------
LENS_FILE = Path("data/lenses.json")
_DEFAULT_LENSES = {
    "maritime trade": {"year_range": (-1000, 1600), "regions": [], "tags": ["trade","naval","maritime","sea","silk-road"]},
    "iron adoption":  {"year_range": (-1300, 600),  "regions": [], "tags": ["iron","metallurgy","tech"]},
    "classical era":  {"year_range": (-800, 500),   "regions": [], "tags": []},
}

def _ensure_lens_file() -> None:
    LENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LENS_FILE.exists():
        with LENS_FILE.open("w", encoding="utf-8") as f:
            json.dump(_DEFAULT_LENSES, f, indent=2)

def _load_lenses() -> dict:
    _ensure_lens_file()
    try:
        with LENS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_lenses(lenses: dict) -> None:
    _ensure_lens_file()
    with LENS_FILE.open("w", encoding="utf-8") as f:
        json.dump(lenses, f, indent=2)

def _apply_lens_to_state(lens: dict) -> None:
    yr = lens.get("year_range", (-500, 500))
    st.session_state["f_year_range"] = (int(yr[0]), int(yr[1]))
    st.session_state["f_regions"] = list(lens.get("regions", []))
    st.session_state["f_tags"]    = list(lens.get("tags", []))

def _on_pick_lens():
    name = st.session_state.get("lens_select")
    lenses = st.session_state.get("_lenses_cache", {})
    lens = lenses.get(name)
    if lens:
        _apply_lens_to_state(lens)

# ---------- Auth gate ----------
def _login_gate():
    if not APP_PASSWORD:
        return
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return
    st.title("🔐 Civilization Explorer")
    st.caption("This app is protected by a simple password gate.")
    pwd = st.text_input("Password", type="password")
    ok = st.button("Enter", type="primary")
    if ok and pwd == APP_PASSWORD:
        st.session_state.authed = True
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    if ok and pwd != APP_PASSWORD:
        st.error("Incorrect password.")
    st.stop()

# ---------- optional: ensure tables exist ----------
def _ensure_tables():
    if engine is None:
        return
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        st.warning(f"Could not auto-create tables: {e}")

# ---------- MAIN APP ----------
def _render_app():
    st.set_page_config(page_title="Civilization Explorer", layout="wide")

    # Login (only if enabled by env) and DB safety
    _login_gate()
    _ensure_tables()

    # ---- Theme lenses (save/load presets) ----
    lenses = _load_lenses()
    st.session_state["_lenses_cache"] = lenses

    with st.sidebar.expander("Theme lenses", expanded=False):
        lens_names = sorted(lenses.keys())
        st.selectbox("Load a lens", options=["—"] + lens_names, key="lens_select", on_change=_on_pick_lens)
        st.divider()
        st.caption("Save current filters as a lens")
        new_name = st.text_input("Lens name", key="lens_name_input", placeholder="e.g., maritime trade 2")
        save_col, del_col = st.columns([1, 1], gap="small")
        with save_col:
            if st.button("Save lens", use_container_width=True, type="primary"):
                if new_name.strip():
                    lenses[new_name.strip()] = {
                        "year_range": st.session_state.get("f_year_range", (-500, 500)),
                        "regions":    st.session_state.get("f_regions", []),
                        "tags":       st.session_state.get("f_tags", []),
                    }
                    _save_lenses(lenses)
                    st.session_state["_lenses_cache"] = lenses
                    st.success(f"Saved lens: {new_name.strip()}", icon="✅")
                else:
                    st.warning("Please enter a lens name.")
        with del_col:
            if lens_names and st.button("Delete selected", use_container_width=True):
                sel = st.session_state.get("lens_select")
                if sel in lenses:
                    del lenses[sel]
                    _save_lenses(lenses)
                    st.session_state["_lenses_cache"] = lenses
                    st.session_state["lens_select"] = "—"
                    st.success(f"Deleted lens: {sel}", icon="✅")
                else:
                    st.info("Pick a lens to delete.")

    # ---- Sidebar controls ----
    with st.sidebar.expander("Tech diffusion (beta)", expanded=False):
        enable_diffusion = st.checkbox("Show diffusion", value=False, key="diff_on")
        diffusion_theme = st.selectbox(
            "Theme", options=["christianity", "ironworking", "trade networks"],
            index=0, key="diff_theme",
            help="Find earliest → later adoptions within your current year window and region filters."
        )

    # Load region/tag options AFTER DB exists (now safe)
    all_regions = get_all_regions()
    all_tags = get_all_tags()

    lens_region_defaults = list(st.session_state.get("f_regions", []))
    lens_tag_defaults    = list(st.session_state.get("f_tags", []))

    region_options = sorted(set(all_regions) | set(lens_region_defaults), key=str.lower)
    tag_options = sorted(set(all_tags) | set([k for k in KIND_TAXONOMY if k not in all_tags]) | set(lens_tag_defaults), key=str.lower)

    st.session_state["f_regions"] = [r for r in lens_region_defaults if r in region_options]
    st.session_state["f_tags"]    = [t for t in lens_tag_defaults    if t in tag_options]

    year_range = st.sidebar.slider("Year range", min_value=-3000, max_value=2000, value=st.session_state.get("f_year_range", (-500, 500)), step=50, key="f_year_range")
    selected_regions = st.sidebar.multiselect("Regions", options=region_options, default=st.session_state.get("f_regions", []), key="f_regions")
    selected_tags = st.sidebar.multiselect("Tags (and kinds)", options=tag_options, default=st.session_state.get("f_tags", []), key="f_tags")

    # ---------- Quiz mode ----------
    with st.sidebar.expander("Quiz mode", expanded=False):
        st.caption("Build a quiz from the current filters.")
        quiz_mode = st.checkbox("Enable quiz mode", value=False, key="quiz_mode")
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=0, step=1, help="Same seed → same quiz.")
        if st.button("Generate 10 questions", use_container_width=True):
            ev_pool, cv_pool = _load_quiz_pool(year_range, selected_regions, _split_csv(selected_tags))
            qz = _generate_quiz(seed, ev_pool, cv_pool)
            if len(qz["questions"]) == 0:
                st.warning("Not enough data under current filters to build a quiz. Widen the year range or clear some filters.")
            else:
                st.session_state["quiz_state"] = qz
                st.success(f"Generated {len(qz['questions'])} question(s).", icon="✅")

    # ---------- Curriculum packs ----------
    with st.sidebar.expander("Curriculum packs", expanded=False):
        st.caption("Bundle the current lens + selected civilizations + (optional) quiz + PNGs into a ZIP.")
        _ev_for_pack = _query_events(year_range, selected_regions, _split_csv(selected_tags))
        _by_civ_pack = _group_events_by_civ(_ev_for_pack)
        _ids_pack = list(_by_civ_pack.keys())
        if _ids_pack:
            with get_session() as _s:
                _civs_pack = _s.exec(select(Civilization).where(Civilization.id.in_(_ids_pack)).order_by(Civilization.name)).all()
        else:
            _civs_pack = []
        _sel_civs = st.multiselect("Include civilizations", options=_civs_pack, default=_civs_pack, format_func=lambda c: c.name, key="pack_sel_civs")
        _incl_quiz   = st.checkbox("Include current quiz (if any)", value=bool(st.session_state.get("quiz_state")))
        _incl_images = st.checkbox("Include PNG snapshots", value=True, help="Requires 'kaleido' for PNG export.")
        _lens_snapshot = {"name": st.session_state.get("lens_select") if st.session_state.get("lens_select") not in (None, "—") else None, "year_range": year_range, "regions": selected_regions, "tags": _split_csv(selected_tags)}
        if st.button("Build curriculum pack (ZIP)", type="primary", use_container_width=True):
            try:
                zip_bytes, zip_name = _build_curriculum_pack_zip(
                    lens_spec=_lens_snapshot, selected_civs=_sel_civs, year_range=year_range,
                    regions=selected_regions, tags=_split_csv(selected_tags),
                    include_quiz=_incl_quiz, include_images=_incl_images,
                )
                st.success("Pack built. Click to download:")
                st.download_button("Download curriculum pack", data=zip_bytes, file_name=zip_name, mime="application/zip", use_container_width=True)
            except Exception as e:
                st.error(f"Could not build pack: {e}")

    # --- Import / Export ---
    with st.sidebar.expander("Import / Export", expanded=False):
        yr = st.session_state.get("f_year_range", year_range)
        regs = st.session_state.get("f_regions", selected_regions)
        tgz  = _split_csv(st.session_state.get("f_tags", selected_tags))
        expo_events = _query_events(yr, regs, tgz)
        st.caption("Export the currently filtered timeline")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("CSV", data=_export_events_csv_bytes(expo_events), file_name="timeline_export.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("JSON", data=_export_events_json_bytes(expo_events), file_name="timeline_export.json", mime="application/json", use_container_width=True)
        with c3:
            fig = _build_timeline_bands(expo_events, yr)
            png = None
            if fig:
                try:
                    png = pio.to_image(fig, format="png", scale=2, engine="kaleido")
                except Exception:
                    png = None
            if png:
                st.download_button("PNG", data=png, file_name="timeline_snapshot.png", mime="image/png", use_container_width=True)
            else:
                st.button("PNG", disabled=True, help="Install 'kaleido' to enable PNG export:\n  poetry add kaleido", use_container_width=True)

        st.divider()
        st.caption("Import events (CSV or JSON) — columns: civ_slug,title,year,kind,summary,tags")
        up = st.file_uploader("Choose a file", type=["csv", "json"], accept_multiple_files=False)
        dry = st.checkbox("Dry run (preview only)", value=True)
        if up:
            raw = up.getvalue()
            rows = _parse_events_csv_bytes(raw) if up.name.lower().endswith(".csv") else _parse_events_json_bytes(raw)
            st.write(f"Parsed {len(rows)} row(s). Showing first 10:")
            st.dataframe(rows[:10])
            if st.button("Apply import"):
                ins, sk, miss = _insert_events_from_rows(rows, dry_run=dry)
                if dry:
                    st.info(f"Dry run → would insert {ins}, skip {sk}, missing civ for {miss}. Uncheck 'Dry run' to apply.")
                else:
                    st.success(f"Imported {ins} new event(s); skipped {sk} duplicate(s); {miss} row(s) had unknown civ_slug.")

    # ---------- Compare or Main ----------
    compare_mode = st.sidebar.checkbox("Compare two civilizations", value=False)

    if compare_mode:
        st.title("Compare civilizations")
        tags_list = _split_csv(selected_tags)
        events = _query_events(year_range, selected_regions, tags_list)
        by_civ = _group_events_by_civ(events)
        civ_ids = list(by_civ.keys())
        if not civ_ids:
            st.info("No civilizations match the current filters. Try widening your year range or clearing tags/regions.")
            st.stop()
        with get_session() as s:
            civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids)).order_by(Civilization.name)).all()

        def find_index(slug_sub: str) -> Optional[int]:
            slug_sub = slug_sub.lower()
            for i, c in enumerate(civs):
                if c.slug and slug_sub in c.slug.lower():
                    return i
            return None

        idx_rome = find_index("roman") or find_index("rome")
        idx_han  = find_index("han")
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

        ev_a = by_civ.get(civ_a.id, [])
        ev_b = by_civ.get(civ_b.id, [])

        def _summ(evts: List[Event]) -> Dict[str, str]:
            if not evts: return {"count": "0", "span": "—"}
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
        cnt_a = _kind_counts(ev_a); cnt_b = _kind_counts(ev_b)
        chart = _build_compare_chart(civ_a.name, cnt_a, civ_b.name, cnt_b)
        st.plotly_chart(chart, use_container_width=True)

        # Downloads
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        w.writerow(["civilization","year","title","kind","tags"])
        for e in ev_a: w.writerow([civ_a.name, e.year, e.title, e.kind or "", e.tags or ""])
        for e in ev_b: w.writerow([civ_b.name, e.year, e.title, e.kind or "", e.tags or ""])
        st.download_button("Download compare events CSV", data=csv_buf.getvalue(), file_name="compare_events.csv", mime="text/csv", use_container_width=True)

        try:
            png_bytes = pio.to_image(chart, format="png", scale=2, engine="kaleido")
            st.download_button("Download compare chart PNG", data=png_bytes, file_name="compare_chart.png", mime="image/png", use_container_width=True)
        except Exception:
            st.button("Download compare chart PNG", disabled=True, help="Install 'kaleido' to enable PNG export:\n  poetry add kaleido", use_container_width=True)

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

        fig = _build_timeline_bands(events, year_range)
        if fig:
            st.subheader("Timeline bands")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events in the selected range to plot.")

        civ_ids = list(by_civ.keys())
        if civ_ids:
            with get_session() as s:
                civs = s.exec(select(Civilization).where(Civilization.id.in_(civ_ids)).order_by(Civilization.name)).all()
        else:
            civs = []

        st.caption(f"{len(civs)} civilization(s) match your filters.")

        map_fig = _build_civ_map(events, by_civ)
        if map_fig:
            st.subheader("Map")
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.info("No mapped civilizations for the current filters.")

        if st.session_state.get("diff_on"):
            theme = st.session_state.get("diff_theme", "christianity")
            adoptions = _first_adoptions(theme, year_range, selected_regions)
            st.subheader(f"Diffusion: {theme.title()}")
            if len(adoptions) < 2:
                st.info("Not enough matching adoptions in this window to draw a path. Try widening the year range, switching regions, or picking another theme.")
            else:
                tfig = _build_diffusion_timeline(adoptions)
                if tfig: st.plotly_chart(tfig, use_container_width=True)
                mfig = _build_diffusion_map(adoptions)
                if mfig: st.plotly_chart(mfig, use_container_width=True)

        st.subheader("Hotspots by century")
        st.caption("Top-quintile centuries are highlighted (based on counts within each series).")
        def _century_start(y: int) -> int: return (math.floor(y / 100.0)) * 100
        def _century_starts_in_range(year_range: Tuple[int,int]) -> List[int]:
            s, e = year_range; s = _century_start(s); e = _century_start(e)
            if e < s: s, e = e, s
            return list(range(s, e + 1, 100))
        def _bin_events_by_century(events: List[Event], year_range: Tuple[int,int]) -> Dict[str, Dict[int, int]]:
            centuries = _century_starts_in_range(year_range)
            if not centuries: return {}
            def _blank_map(): return {c: 0 for c in centuries}
            per_region: Dict[str, Dict[int,int]] = {}; all_map = _blank_map()
            for e in events:
                c = e.civilization
                if not c: continue
                cs = _century_start(e.year)
                if cs < centuries[0] or cs > centuries[-1]: continue
                region = (c.region or "Unknown").strip()
                per_region.setdefault(region, _blank_map()); per_region[region][cs] += 1
                all_map[cs] += 1
            if any(v > 0 for v in all_map.values()):
                per_region = {"All (visible)": all_map, **per_region}
            return per_region
        def _quintile_threshold(counts: List[int]) -> Optional[int]:
            positives = [c for c in counts if c > 0]
            if not positives: return None
            positives.sort(); q_idx = max(0, math.ceil(0.8 * len(positives)) - 1)
            return positives[q_idx]
        def _build_region_sparkline(title: str, series: Dict[int,int]) -> Optional[go.Figure]:
            if not series: return None
            xs = sorted(series.keys()); ys = [series[x] for x in xs]; thr = _quintile_threshold(ys)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False))
            if thr is not None:
                hx = [x for x, y in zip(xs, ys) if y >= thr]; hy = [y for y in ys if y >= thr]
                fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=9, color="#F39C12", line=dict(width=0.5)), name="hotspot",
                                         hovertemplate=f"<b>{title}</b><br>%{{x}} → %{{y}} events<extra></extra>", showlegend=False))
            fig.update_layout(height=120, margin=dict(l=10,r=10,t=8,b=8),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, rangemode="tozero"),
                              title=dict(text=title, x=0.02, y=0.95, xanchor="left", yanchor="top", font=dict(size=12)))
            return fig
        binned = _bin_events_by_century(events, year_range)
        if not binned:
            st.info("No events in this window to analyze.")
        else:
            regions = list(binned.keys()); cols_per_row = 3
            for i in range(0, len(regions), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, region in zip(cols, regions[i:i+cols_per_row]):
                    with col:
                        fig = _build_region_sparkline(region, binned[region])
                        if fig: st.plotly_chart(fig, use_container_width=True)

        st.subheader("Narrative summary")
        st.caption("Click to generate an extractive summary for the current filters (200–300 words).")
        if st.button("Generate summary"):
            top_ev = _pick_top_events(events, per_civ_cap=3, total_cap=14)
            draft = _make_narrative(top_ev, year_range, selected_regions, _split_csv(selected_tags))
            st.session_state["narrative_text"] = draft
        st.text_area("Edit your summary", value=st.session_state.get("narrative_text", ""), height=260, key="narrative_text",
                     help="This is your editable draft. Tweak as needed, then copy it into your notes or book.")

        cols_per_row = 3
        for i in range(0, len(civs), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, civ in zip(cols, civs[i:i+cols_per_row]):
                with col:
                    with st.container(border=True):
                        st.subheader(civ.name)
                        st.text(civ.region or "")
                        sample = by_civ.get(civ.id, [])[:3]
                        if sample:
                            st.write("Sample events:")
                            for e in sample:
                                st.write(f"- {e.year}: {e.title}")
                        st.button("View details", key=f"btn_{civ.id}", on_click=lambda cid=civ.id: st.session_state.update(selected_civ_id=cid))

    else:
        with get_session() as s:
            civ = s.exec(
                select(Civilization).where(Civilization.id == st.session_state.selected_civ_id).options(selectinload(Civilization.sources))
            ).one_or_none()

            if civ:
                stmt = (
                    select(Event)
                    .where(Event.civilization_id == civ.id, Event.year >= year_range[0], Event.year <= year_range[1])
                    .options(selectinload(Event.civilization), selectinload(Event.sources))
                    .order_by(Event.year)
                )
                if selected_tags:
                    stmt = stmt.where(or_(*[Event.tags.ilike(f"%{t}%") for t in _split_csv(selected_tags)]))
                events = s.exec(stmt).all()
            else:
                events = []

            civ_data = None
            if civ:
                civ_data = {
                    "id": civ.id, "name": civ.name, "region": civ.region,
                    "start_year": civ.start_year, "end_year": civ.end_year,
                    "sources": [{"key": src.key, "title": src.title, "url": src.url} for src in civ.sources],
                }
            events_data = [{
                "id": ev.id, "year": ev.year, "title": ev.title, "summary": ev.summary,
                "kind": ev.kind, "tags": ev.tags,
                "sources": [{"key": src.key, "title": src.title, "url": src.url} for src in ev.sources],
            } for ev in events]

        if not civ_data:
            st.warning("Civilization not found.")
            st.session_state.selected_civ_id = None
        else:
            st.button("← Back to list", on_click=lambda: st.session_state.update(selected_civ_id=None))
            st.title(civ_data["name"])
            st.caption(civ_data["region"] or "")

            if civ_data["sources"]:
                st.subheader("Sources for this civilization")
                for src in civ_data["sources"]:
                    label = src["title"] or src["key"]
                    st.markdown(f"- [{label}]({src['url']})" if src["url"] else f"- {label}")

            st.divider()
            fig = _build_timeline_bands(events, year_range)
            if fig:
                st.subheader("Timeline for this civilization")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.write(f"{len(events)} matching event(s) for {civ_data['name']}:")
            for ev, evd in zip(events, events_data):
                with st.container(border=True):
                    st.write(f"**{evd['year']} — {evd['title']}**")
                    if evd["summary"]:
                        st.write(evd["summary"])
                    meta = ", ".join(filter(None, [evd["kind"], evd["tags"]]))
                    if meta:
                        st.caption(meta)
                    if evd["sources"]:
                        st.markdown("_Sources_:")
                        for src in evd["sources"]:
                            label = src["title"] or src["key"]
                            st.markdown(f"  - [{label}]({src['url']})" if src["url"] else f"  - {label}")

            st.subheader("Neighbors in time")
            st.caption(f"Overlaps computed within the current window: {_fmt_year(year_range[0])} to {_fmt_year(year_range[1])}.")
            nbrs = _neighbors_by_overlap(type("CivLite", (), {"id": civ_data["id"], "name": civ_data["name"], "start_year": civ_data["start_year"], "end_year": civ_data["end_year"]})(), year_range)
            if not nbrs:
                st.info("No overlapping civilizations in this window.")
            else:
                for nb in nbrs:
                    st.write(f"- **{nb['name']}** — {nb['overlap']} yrs ({_fmt_year(nb['ov_start'])} → {_fmt_year(nb['ov_end'])})")

            # Sequence hints
            hints = _sequence_hints(type("CivLite", (), {"id": civ_data["id"], "name": civ_data["name"], "start_year": civ_data["start_year"], "end_year": civ_data["end_year"]})(), events, year_range, neighbors=nbrs, delta=50)
            c1, c2 = st.columns([1, 8])
            with c1:
                st.button("ⓘ", help=("Sequence hints surface events in overlapping civilizations that occur within ±50 years of this civilization’s events.\nThey show temporal sequence, not causation."), disabled=True)
            with c2:
                st.metric("Sequence hints", len(hints))
            if not hints:
                st.caption("No near-in-time neighbor changes in this window.")
            else:
                grouped: Dict[str, List[Dict]] = {}
                for h in hints:
                    grouped.setdefault(h["neighbor_name"], []).append(h)
                for neigh, items in sorted(grouped.items(), key=lambda kv: kv[0]):
                    st.write(f"**{neigh}**")
                    for h in items:
                        st.write(f"- {h['year']}: {h['title']}  ·  {h['kind']}  ·  Δ{h['distance']} yrs")

if __name__ == "__main__":
    _render_app()
