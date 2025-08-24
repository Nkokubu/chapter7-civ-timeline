from typing import Optional
from pydantic import BaseModel

class CivilizationBrief(BaseModel):
    id: int
    slug: str
    name: str

class EventOut(BaseModel):
    id: int
    title: str
    year: int
    kind: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[str] = None
    civilization: CivilizationBrief