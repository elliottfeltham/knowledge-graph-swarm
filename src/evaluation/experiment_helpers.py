import copy
import random
from dataclasses import dataclass
from statistics import mean

from agents import GraphUpdateConfig
from agents.graph_statistics import calculate_graph_statistics
from agents.graph_update import update_graph_after_retrieval
from common import faiss_retrieve, mean_reciprocal_rank, recall_at_k
from retrieval.agent_graph_retrieval import agent_graph_retrieve


@dataclass
class EvaluationContext:
    query_embeddings: dict
    qrels: dict[str, set[str]]
    index: object
    document_ids: list[str]
    initial_agents: dict
    top_k: int


def split_query_groups(
    queries: list[dict],
    seed: int,
    adaptation_count: int,
    validation_count: int,
    evaluation_count: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    required_queries = adaptation_count + validation_count + evaluation_count

    if len(queries) < required_queries:
        raise ValueError(
            f"Need {required_queries} queries but only found {len(queries)}"
        )

    shuffled_queries = queries.copy()
    random.Random(seed).shuffle(shuffled_queries)
    validation_start = adaptation_count
    evaluation_start = validation_start + validation_count
    evaluation_end = evaluation_start + evaluation_count

    return (
        shuffled_queries[:validation_start],
        shuffled_queries[validation_start:evaluation_start],
        shuffled_queries[evaluation_start:evaluation_end],
    )


def calculate_metrics(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 5),
        "recall_at_10": recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10),
        "mrr": mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
    }


def compare_query_metrics(agent_metrics: dict, faiss_metrics: dict) -> str:
    agent_score = (agent_metrics["recall_at_10"], agent_metrics["mrr"])
    faiss_score = (faiss_metrics["recall_at_10"], faiss_metrics["mrr"])

    if agent_score > faiss_score:
        return "improved"
    if agent_score < faiss_score:
        return "worsened"
    return "unchanged"


def summarise_metrics(metric_rows: list[dict]) -> dict:
    return {
        "queries_evaluated": len(metric_rows),
        "recall_at_5": mean(row["recall_at_5"] for row in metric_rows),
        "recall_at_10": mean(row["recall_at_10"] for row in metric_rows),
        "mrr": mean(row["mrr"] for row in metric_rows),
    }


def summarise_agent_rows(rows: list[dict]) -> dict:
    return {
        **summarise_metrics([row["metrics"] for row in rows]),
        "queries_improved": sum(row["outcome"] == "improved" for row in rows),
        "queries_unchanged": sum(row["outcome"] == "unchanged" for row in rows),
        "queries_worsened": sum(row["outcome"] == "worsened" for row in rows),
    }


def calculate_faiss_metrics(
    queries: list[dict],
    context: EvaluationContext,
) -> dict[str, dict]:
    results = {}

    for query in queries:
        query_id = query["query_id"]
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in faiss_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query_id],
                document_ids=context.document_ids,
                top_k=context.top_k,
            )
        ]
        results[query_id] = calculate_metrics(
            retrieved_doc_ids,
            context.qrels.get(query_id, set()),
        )

    return results


def evaluate_agent_graph(
    queries: list[dict],
    context: EvaluationContext,
    agents,
    seed_k: int,
    graph_weight: float,
    faiss_metrics: dict[str, dict],
) -> list[dict]:
    rows = []

    for query in queries:
        query_id = query["query_id"]
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query_id],
                document_ids=context.document_ids,
                agents=agents,
                top_k=context.top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]
        metrics = calculate_metrics(
            retrieved_doc_ids,
            context.qrels.get(query_id, set()),
        )
        rows.append(
            {
                "query_id": query_id,
                "metrics": metrics,
                "outcome": compare_query_metrics(metrics, faiss_metrics[query_id]),
            }
        )

    return rows


def adapt_graph(
    queries: list[dict],
    context: EvaluationContext,
    agents,
    seed_k: int,
    graph_weight: float,
    update_k: int | None = None,
    update_config: GraphUpdateConfig | None = None,
) -> dict:
    totals = {
        "strengthened_edges": 0,
        "created_edges": 0,
        "pruned_edges": 0,
    }
    if update_config is None:
        update_config = GraphUpdateConfig(
            enable_pruning=False,
            update_k=update_k,
        )

    for query in queries:
        retrieved_doc_ids = [
            doc_id
            for doc_id, _ in agent_graph_retrieve(
                index=context.index,
                query_embedding=context.query_embeddings[query["query_id"]],
                document_ids=context.document_ids,
                agents=agents,
                top_k=context.top_k,
                seed_k=seed_k,
                graph_weight=graph_weight,
            )
        ]
        update_summary = update_graph_after_retrieval(
            agents=agents,
            retrieved_doc_ids=retrieved_doc_ids,
            config=update_config,
        )

        for key in totals:
            totals[key] += update_summary[key]

    return totals


def aggregate_seed_summaries(seed_results: list[dict], graph_state: str) -> dict:
    summaries = [result[graph_state] for result in seed_results]

    return {
        "queries_evaluated": sum(row["queries_evaluated"] for row in summaries),
        "recall_at_5": mean(row["recall_at_5"] for row in summaries),
        "recall_at_10": mean(row["recall_at_10"] for row in summaries),
        "mrr": mean(row["mrr"] for row in summaries),
        "queries_improved": sum(row["queries_improved"] for row in summaries),
        "queries_unchanged": sum(row["queries_unchanged"] for row in summaries),
        "queries_worsened": sum(row["queries_worsened"] for row in summaries),
    }


def evaluate_configuration(
    context: EvaluationContext,
    query_splits: dict,
    query_group: int,
    seed_k: int,
    graph_weight: float,
    update_k: int | None = None,
    update_config: GraphUpdateConfig | None = None,
) -> dict:
    seed_results = []
    all_faiss_metrics = []

    for seed, query_groups in query_splits.items():
        adaptation_queries = query_groups[0]
        evaluation_queries = query_groups[query_group]
        faiss_metrics = calculate_faiss_metrics(evaluation_queries, context)
        initial_rows = evaluate_agent_graph(
            evaluation_queries,
            context,
            context.initial_agents,
            seed_k,
            graph_weight,
            faiss_metrics,
        )
        adapted_agents = copy.deepcopy(context.initial_agents)
        graph_before = calculate_graph_statistics(adapted_agents)
        updates = adapt_graph(
            adaptation_queries,
            context,
            adapted_agents,
            seed_k,
            graph_weight,
            update_k,
            update_config,
        )
        graph_after = calculate_graph_statistics(adapted_agents)
        adapted_rows = evaluate_agent_graph(
            evaluation_queries,
            context,
            adapted_agents,
            seed_k,
            graph_weight,
            faiss_metrics,
        )

        all_faiss_metrics.extend(faiss_metrics.values())
        seed_results.append(
            {
                "seed": seed,
                "updates": updates,
                "graph_before": graph_before,
                "graph_after": graph_after,
                "initial": summarise_agent_rows(initial_rows),
                "adapted": summarise_agent_rows(adapted_rows),
            }
        )

    return {
        "faiss": summarise_metrics(all_faiss_metrics),
        "initial": aggregate_seed_summaries(seed_results, "initial"),
        "adapted": aggregate_seed_summaries(seed_results, "adapted"),
        "seed_results": seed_results,
    }
