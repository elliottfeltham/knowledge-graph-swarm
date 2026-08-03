from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1)
    seed_k: int = Field(default=5, ge=1)
    graph_weight: float = Field(default=0.3, ge=0)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")

        return value


class SearchResult(BaseModel):
    doc_id: str
    title: str
    score: float
    text_preview: str


class SearchResponse(BaseModel):
    query: str
    adaptive: bool
    latency_ms: float
    results: list[SearchResult]


class AdaptiveSearchResponse(SearchResponse):
    update_summary: dict[str, Any]
