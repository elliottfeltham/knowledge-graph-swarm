import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agents import DocumentAgent, GraphUpdateConfig, load_agents, save_agents
from agents.graph_statistics import calculate_graph_statistics
from agents.graph_update import update_graph_after_retrieval
from common import (
    MODEL_NAME,
    build_faiss_index,
    encode_query,
    load_documents,
    load_embeddings,
)
from graph.initialize_agent_graph import build_initial_agent_graph
from retrieval.agent_graph_retrieval import AGENT_GRAPH_PATH, agent_graph_retrieve


@dataclass
class AgentGraphRuntime:
    documents: dict[str, dict]
    embeddings: np.ndarray
    document_ids: list[str]
    index: faiss.Index
    model: SentenceTransformer
    agents: dict[str, DocumentAgent]
    agent_graph_path: Path

    @classmethod
    def load(cls, agent_graph_path: Path = AGENT_GRAPH_PATH) -> "AgentGraphRuntime":
        documents = load_documents()
        embeddings, document_ids = load_embeddings()
        index = build_faiss_index(embeddings)
        model = SentenceTransformer(MODEL_NAME)
        agents = load_agents(agent_graph_path)

        return cls(
            documents=documents,
            embeddings=embeddings,
            document_ids=document_ids,
            index=index,
            model=model,
            agents=agents,
            agent_graph_path=agent_graph_path,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        seed_k: int = 5,
        graph_weight: float = 0.3,
    ) -> tuple[list[dict], float]:
        start_time = time.perf_counter()

        query_embedding = encode_query(self.model, query)

        ranked_results = agent_graph_retrieve(
            index=self.index,
            query_embedding=query_embedding,
            document_ids=self.document_ids,
            agents=self.agents,
            top_k=top_k,
            seed_k=seed_k,
            graph_weight=graph_weight,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return self.format_results(ranked_results), latency_ms

    def adaptive_search(
        self,
        query: str,
        top_k: int = 10,
        seed_k: int = 5,
        graph_weight: float = 0.3,
        update_config: GraphUpdateConfig | None = None,
    ) -> tuple[list[dict], float, dict]:
        results, latency_ms = self.search(
            query=query,
            top_k=top_k,
            seed_k=seed_k,
            graph_weight=graph_weight,
        )
        retrieved_doc_ids = [result["doc_id"] for result in results]

        update_summary = update_graph_after_retrieval(
            agents=self.agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=update_config,
        )
        self.save_graph()

        return results, latency_ms, update_summary

    def save_graph(self) -> None:
        save_agents(self.agent_graph_path, self.agents)

    def graph_statistics(self) -> dict:
        return calculate_graph_statistics(self.agents)

    def reset_graph(
        self,
        neighbours_per_doc: int = 5,
        initial_weight_scale: float = 0.2,
    ) -> dict:
        self.agents = build_initial_agent_graph(
            neighbours_per_doc=neighbours_per_doc,
            initial_weight_scale=initial_weight_scale,
        )
        self.save_graph()

        return self.graph_statistics()

    def format_results(self, ranked_results: list[tuple[str, float]]) -> list[dict]:
        results = []

        for doc_id, score in ranked_results:
            document = self.documents[doc_id]

            results.append(
                {
                    "doc_id": doc_id,
                    "title": document["title"],
                    "score": float(score),
                    "text_preview": document["text"][:300],
                }
            )

        return results
