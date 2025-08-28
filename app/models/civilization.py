
from typing import Optional, List, TYPE_CHECKING      # ← List is imported
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from .event import Event

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

    # MUST be typing.List[...] (capital L), not built-in list[...]
    events: List["Event"] = Relationship(back_populates="civilization")

    # NEW: centroid (nullable)
    latitude: Optional[float] = Field(default=None, description="Centroid latitude")
    longitude: Optional[float] = Field(default=None, description="Centroid longitude")

    # relationships...
    events: List["Event"] = Relationship(back_populates="civilization")
