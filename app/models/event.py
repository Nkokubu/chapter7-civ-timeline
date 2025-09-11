from typing import Optional, List, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from .source import EventSourceLink  # link model only

if TYPE_CHECKING:
    from .civilization import Civilization
    from .source import Source

class Event(SQLModel, table=True):
    __tablename__ = "events"

    id: Optional[int] = Field(default=None, primary_key=True)
    civilization_id: int = Field(foreign_key="civilizations.id", index=True)
    title: str
    year: int
    kind: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[str] = None

    civilization: "Civilization" = Relationship(back_populates="events")
    sources: List["Source"] = Relationship(back_populates="events", link_model=EventSourceLink)






