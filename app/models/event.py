
from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from .civilization import Civilization

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


