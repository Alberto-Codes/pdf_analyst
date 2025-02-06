from typing import List

from pydantic import BaseModel, Field


class Director(BaseModel):
    name: str = Field(..., description="Director's full name")
    age: int = Field(None, description="Director's age (if available)")
    title: str = Field(..., description="Director's title")


class Directors(BaseModel):
    directors: List[Director]
