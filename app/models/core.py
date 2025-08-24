from __future__ import annotations
from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel

# --- link model FIRST so it's in scope below ---
class EventTag(SQLModel, table=True):
    event_id: int = Field(foreign_key="event.id", primary_key=True)
    tag_id: int = Field(foreign_key="tag.id", primary_key=True)

class Civilization(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    region: str = Field(index=True)
    start_year: int = Field(index=True, description="BCE negative")
    end_year: Optional[int] = Field(default=None, index=True, description="BCE negative; None if ongoing")

    events: List["Event"] = Relationship(back_populates="civilization")

class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    civilization_id: int = Field(foreign_key="civilization.id", index=True)

    title: str = Field(index=True)
    year: int = Field(index=True, description="BCE negative")
    description: Optional[str] = None
    kind: Optional[str] = Field(default=None, description="war/dynasty/tech/culture/economy/religion")

    civilization: Optional[Civilization] = Relationship(back_populates="events")
    tags: List["Tag"] = Relationship(back_populates="events", link_model=EventTag)

class Tag(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

    events: List[Event] = Relationship(back_populates="tags", link_model=EventTag)
