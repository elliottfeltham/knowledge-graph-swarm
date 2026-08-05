from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AdaptiveSearchResponse,
    DocumentDetailResponse,
    GraphResponse,
    SearchRequest,
    SearchResponse,
    SubgraphRequest,
)
from retrieval.agent_graph_runtime import AgentGraphRuntime


runtime: AgentGraphRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime

    runtime = AgentGraphRuntime.load()

    yield


app = FastAPI(
    title="Knowledge Graph Swarm API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "runtime_loaded": runtime is not None,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    results, latency_ms = runtime.search(
        query=request.query,
        top_k=request.top_k,
        seed_k=request.seed_k,
        graph_weight=request.graph_weight,
    )

    return SearchResponse(
        query=request.query,
        adaptive=False,
        latency_ms=latency_ms,
        results=results,
    )


@app.post("/adaptive-search", response_model=AdaptiveSearchResponse)
def adaptive_search(request: SearchRequest) -> AdaptiveSearchResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    results, latency_ms, update_summary = runtime.adaptive_search(
        query=request.query,
        top_k=request.top_k,
        seed_k=request.seed_k,
        graph_weight=request.graph_weight,
    )

    return AdaptiveSearchResponse(
        query=request.query,
        adaptive=True,
        latency_ms=latency_ms,
        results=results,
        update_summary=update_summary,
    )


@app.get("/graph/stats")
def graph_stats() -> dict:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    return runtime.graph_statistics()


@app.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def document_detail(doc_id: str) -> DocumentDetailResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    try:
        document = runtime.document_data(doc_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return DocumentDetailResponse(**document)


@app.get("/graph", response_model=GraphResponse)
def full_graph() -> GraphResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    return GraphResponse(**runtime.graph_data())


@app.post("/graph/subgraph", response_model=GraphResponse)
def graph_subgraph(request: SubgraphRequest) -> GraphResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    try:
        graph_data = runtime.subgraph_data(
            document_ids=request.document_ids,
            depth=request.depth,
            max_nodes=request.max_nodes,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return GraphResponse(**graph_data)


@app.post("/graph/reset")
def reset_graph() -> dict:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime is not loaded")

    graph_statistics = runtime.reset_graph()

    return {
        "status": "reset",
        "graph_statistics": graph_statistics,
    }
