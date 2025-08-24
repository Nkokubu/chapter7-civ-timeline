# app/main.py
from typing import List, Optional, Generator

from fastapi import FastAPI, Depends, Query, HTTPException
from sqlmodel import SQLModel, Session, select
from sqlalchemy import and_, or_

from app.db.session import get_session, engine
from app.models.civilization import Civilization
from app.models.event import Event
from sqlalchemy.orm import selectinload
from app.models.schemas import EventOut, CivilizationBrief

# FastAPI instance (uvicorn target is "app.main:api")
api = FastAPI(title="Civilization Timeline API")

# Ensure tables exist (safe if you also use Alembic; it's idempotent)
@api.on_event("startup")
def _create_tables() -> None:
    SQLModel.metadata.create_all(engine)

# Dependency: yield a DB session per request
def db() -> Generator[Session, None, None]:
    with get_session() as s:
        yield s

@api.get("/civs", response_model=List[Civilization])
def list_civilizations(
    region: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    s: Session = Depends(db),
) -> List[Civilization]:
    stmt = select(Civilization)
    if region:
        stmt = stmt.where(Civilization.region == region)
    stmt = stmt.order_by(Civilization.name).limit(limit)
    return list(s.exec(stmt).all())

@api.get("/events", response_model=List[EventOut])
def list_events(
    start: Optional[int] = Query(default=None, description="Earliest year (BCE negative)"),
    end: Optional[int] = Query(default=None, description="Latest year"),
    civ: Optional[str] = Query(default=None, description="Civilization slug OR name (case-insensitive)"),
    kind: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    s: Session = Depends(db),
) -> List[EventOut]:
    # eager-load the civilization relationship so we can read its name/slug
    stmt = select(Event).options(selectinload(Event.civilization))
    conditions = []

    if start is not None:
        conditions.append(Event.year >= start)
    if end is not None:
        conditions.append(Event.year <= end)
    if kind:
        conditions.append(Event.kind == kind)

    # allow civ filter by slug OR by name (case-insensitive)
    if civ:
        # join only when filtering by civilization fields
        stmt = stmt.join(Civilization)
        conditions.append(or_(Civilization.slug == civ, Civilization.name.ilike(f"%{civ}%")))

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(Event.year).limit(limit)

    events = s.exec(stmt).all()

    # map ORM objects to nice response dicts
    out: List[EventOut] = []
    for e in events:
        c = e.civilization  # loaded via selectinload
        # safety: skip if somehow missing
        if c is None:
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