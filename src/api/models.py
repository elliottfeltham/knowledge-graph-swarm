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


class SubgraphRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    depth: int = Field(default=1, ge=0, le=3)
    max_nodes: int = Field(default=250, ge=1, le=1000)


class GraphNode(BaseModel):
    id: str
    title: str
    degree: int


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    activation_count: int


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_nodes: int
    total_edges: int
    truncated: bool


class DocumentDetailResponse(BaseModel):
    doc_id: str
    title: str
    text_preview: str
    degree: int
