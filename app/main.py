# app/main.py
from __future__ import annotations

from typing import Generator, List, Optional

from fastapi import Depends, FastAPI, Query
from sqlalchemy import and_, or_
from sqlalchemy.orm import selectinload
from sqlmodel import SQLModel, Session, select

from app.db.session import engine, get_session
from app.models.civilization import Civilization
from app.models.event import Event
from app.models.schemas import CivilizationBrief, EventOut

# FastAPI instance (uvicorn target is "app.main:api")
api = FastAPI(title="Civilization Timeline API")


# --- helpers -----------------------------------------------------------------

def db() -> Generator[Session, None, None]:
    """Yield a DB session per request."""
    with get_session() as s:
        yield s


def _split_csv(v: Optional[str]) -> List[str]:
    """Split 'a,b,c' into ['a','b','c'], trimming blanks. Returns [] if None/empty."""
    if not v:
        return []
    return [p.strip() for p in v.split(",") if p.strip()]


# Ensure tables exist (safe alongside Alembic in dev)
@api.on_event("startup")
def _create_tables() -> None:
    SQLModel.metadata.create_all(engine)


# --- routes: civilizations ----------------------------------------------------

@api.get("/civs", response_model=List[Civilization])
def list_civilizations(
    region: Optional[str] = Query(
        default=None,
        description="One or more region terms (comma-separated). Partial, case-insensitive.",
    ),
    name: Optional[str] = Query(
        default=None,
        description="Partial match on civilization name or slug.",
    ),
    limit: int = Query(default=1000, ge=1, le=10000),
    s: Session = Depends(db),
) -> List[Civilization]:
    stmt = select(Civilization)

    regions = _split_csv(region)
    if regions:
        stmt = stmt.where(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))

    if name:
        stmt = stmt.where(
            or_(
                Civilization.name.ilike(f"%{name}%"),
                Civilization.slug.ilike(f"%{name}%"),
            )
        )

    stmt = stmt.order_by(Civilization.name).limit(limit)
    return list(s.exec(stmt).all())


# --- routes: events -----------------------------------------------------------

@api.get("/events", response_model=List[EventOut])
def list_events(
    start: Optional[int] = Query(default=None, description="Earliest year (BCE negative)"),
    end: Optional[int] = Query(default=None, description="Latest year (CE positive)"),
    civ: Optional[str] = Query(
        default=None,
        description="Filter by civilization slug OR name (partial OK).",
    ),
    region: Optional[str] = Query(
        default=None,
        description="One or more region terms (comma-separated). Partial, case-insensitive.",
    ),
    tags: Optional[str] = Query(
        default=None,
        description="One or more tag terms (comma-separated). Partial, case-insensitive.",
    ),
    kind: Optional[str] = Query(default=None, description="Exact kind (e.g., 'war', 'tech')."),
    limit: int = Query(default=1000, ge=1, le=10000),
    s: Session = Depends(db),
) -> List[EventOut]:
    # Base select with eager-load so we can include civ details in response
    stmt = select(Event).options(selectinload(Event.civilization))
    conds = []

    if start is not None:
        conds.append(Event.year >= start)
    if end is not None:
        conds.append(Event.year <= end)
    if kind:
        conds.append(Event.kind == kind)

    # Join to Civilization only if needed for civ/region filters
    need_join = False

    if civ:
        need_join = True
        conds.append(
            or_(
                Civilization.slug.ilike(f"%{civ}%"),
                Civilization.name.ilike(f"%{civ}%"),
            )
        )

    regions = _split_csv(region)
    if regions:
        need_join = True
        conds.append(or_(*[Civilization.region.ilike(f"%{r}%") for r in regions]))

    if need_join:
        stmt = stmt.join(Civilization)

    tag_terms = _split_csv(tags)
    if tag_terms:
        conds.append(or_(*[Event.tags.ilike(f"%{t}%") for t in tag_terms]))

    if conds:
        stmt = stmt.where(and_(*conds))

    stmt = stmt.order_by(Event.year).limit(limit)
    rows = s.exec(stmt).all()

    out: List[EventOut] = []
    for e in rows:
        c = e.civilization
        if not c:
            # Shouldn't happen because of foreign-key + selectinload, but be safe.
            continue
        out.append(
            EventOut(
                id=e.id,
                title=e.title,
                year=e.year,
                kind=e.kind,
                summary=e.summary,
                tags=e.tags,
                civilization=CivilizationBrief(id=c.id, slug=c.slug, name=c.name),
            )
        )
    return out


# --- meta/discovery routes ----------------------------------------------------

@api.get("/meta/regions", response_model=List[str])
def list_regions(s: Session = Depends(db)) -> List[str]:
    rows = s.exec(select(Civilization.region).distinct().order_by(Civilization.region)).all()
    return [r[0] if isinstance(r, tuple) else r for r in rows]


@api.get("/meta/tags", response_model=List[str])
def list_tags(s: Session = Depends(db)) -> List[str]:
    rows = s.exec(select(Event.tags).where(Event.tags.is_not(None))).all()
    seen, out = set(), []
    for r in rows:
        val = r[0] if isinstance(r, tuple) else r
        for t in (val or "").split(","):
            tag = t.strip()
            key = tag.lower()
            if tag and key not in seen:
                seen.add(key)
                out.append(tag)
    return sorted(out, key=str.lower)


@api.get("/meta/civs/slugs", response_model=List[str])
def list_civ_slugs(s: Session = Depends(db)) -> List[str]:
    rows = s.exec(select(Civilization.slug).order_by(Civilization.name)).all()
    return [r[0] if isinstance(r, tuple) else r for r in rows]