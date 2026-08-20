from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(max_length=1_000)
    top_k: int = Field(default=10, ge=1, le=100)
    seed_k: int = Field(default=3, ge=1, le=100)
    graph_weight: float = Field(default=0.05, ge=0, le=1)
    update_k: int = Field(default=3, ge=2, le=100)
    strengthen_amount: float = Field(default=0.01, gt=0, le=1)
    decay_rate: float = Field(default=0.995, ge=0, le=1)
    enable_pruning: bool = False

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")

        return value.strip()


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


class GraphEdgeUpdate(BaseModel):
    source: str
    target: str
    previous_weight: float
    new_weight: float | None
    change_type: Literal["strengthened", "created", "decayed", "pruned"]


class GraphUpdateSummary(BaseModel):
    retrieved_documents: int
    updated_documents: int
    strengthened_edges: int
    created_edges: int
    decayed_edges: int
    pruned_edges: int
    edge_updates: list[GraphEdgeUpdate]


class AdaptiveSearchResponse(SearchResponse):
    update_summary: GraphUpdateSummary


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
