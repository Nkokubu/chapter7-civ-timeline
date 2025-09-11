from typing import Optional, List, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

# Do NOT import Civilization/Event at runtime to avoid cycles
if TYPE_CHECKING:
    from .civilization import Civilization
    from .event import Event

class CivSourceLink(SQLModel, table=True):
    __tablename__ = "civ_source_link"
    civilization_id: int = Field(foreign_key="civilizations.id", primary_key=True)
    source_id: int = Field(foreign_key="sources.id", primary_key=True)

class EventSourceLink(SQLModel, table=True):
    __tablename__ = "event_source_link"
    event_id: int = Field(foreign_key="events.id", primary_key=True)
    source_id: int = Field(foreign_key="sources.id", primary_key=True)

class Source(SQLModel, table=True):
    __tablename__ = "sources"

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(index=True, unique=True)
    title: str
    author: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    note: Optional[str] = None

    # The annotations are the ONLY place SQLModel learns the target model.
    civilizations: List["Civilization"] = Relationship(
        back_populates="sources",
        link_model=CivSourceLink,
    )
    events: List["Event"] = Relationship(
        back_populates="sources",
        link_model=EventSourceLink,
    )


















