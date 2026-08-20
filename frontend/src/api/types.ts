export interface HealthResponse {
  status: string;
  runtime_loaded: boolean;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  seed_k?: number;
  graph_weight?: number;
  update_k?: number;
  strengthen_amount?: number;
  decay_rate?: number;
  enable_pruning?: boolean;
}

export interface SearchResult {
  doc_id: string;
  title: string;
  score: number;
  text_preview: string;
}

export interface SearchResponse {
  query: string;
  adaptive: boolean;
  latency_ms: number;
  results: SearchResult[];
}

export type GraphEdgeChangeType =
  | "strengthened"
  | "created"
  | "decayed"
  | "pruned";

export interface GraphEdgeUpdate {
  source: string;
  target: string;
  previous_weight: number;
  new_weight: number | null;
  change_type: GraphEdgeChangeType;
}

export interface GraphUpdateSummary {
  retrieved_documents: number;
  updated_documents: number;
  strengthened_edges: number;
  created_edges: number;
  decayed_edges: number;
  pruned_edges: number;
  edge_updates: GraphEdgeUpdate[];
}

export interface AdaptiveSearchResponse extends SearchResponse {
  update_summary: GraphUpdateSummary;
}

export interface GraphStatistics {
  agents: number;
  undirected_edges: number;
  average_degree: number;
  average_edge_weight: number;
  min_edge_weight: number;
  max_edge_weight: number;
}

export interface ResetGraphResponse {
  status: string;
  graph_statistics: GraphStatistics;
}

export interface SubgraphRequest {
  document_ids?: string[];
  depth?: number;
  max_nodes?: number;
}

export interface GraphNode {
  id: string;
  title: string;
  degree: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
}

export interface DocumentDetail {
  doc_id: string;
  title: string;
  text_preview: string;
  degree: number;
}
