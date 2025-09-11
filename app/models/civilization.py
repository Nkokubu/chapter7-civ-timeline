from typing import Optional, List, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship
from .source import CivSourceLink  # link model only

if TYPE_CHECKING:
    from .event import Event
    from .source import Source

class Civilization(SQLModel, table=True):
    __tablename__ = "civilizations"

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True)
    name: str
    region: str
    start_year: int
    end_year: int
    lat: Optional[float] = None
    lon: Optional[float] = None

    events: List["Event"] = Relationship(back_populates="civilization")
    sources: List["Source"] = Relationship(back_populates="civilizations", link_model=CivSourceLink)


