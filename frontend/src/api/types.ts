export interface HealthResponse {
  status: string;
  runtime_loaded: boolean;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  seed_k?: number;
  graph_weight?: number;
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

export interface GraphUpdateSummary {
  retrieved_documents: number;
  strengthened_edges: number;
  pruned_edges: number;
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
  activation_count: number;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
  truncated: boolean;
}
