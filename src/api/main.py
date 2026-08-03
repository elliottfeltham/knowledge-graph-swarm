from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models import AdaptiveSearchResponse, SearchRequest, SearchResponse
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
