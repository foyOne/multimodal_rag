from pydantic import BaseModel, Field
from documents.document import Document


class RankedDocument(BaseModel):
    rank: int
    score: float
    doc: Document = Field(default_factory=Document)
