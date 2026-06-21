from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    data: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
