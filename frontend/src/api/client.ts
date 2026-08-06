import type {
  AdaptiveSearchResponse,
  DocumentDetail,
  GraphResponse,
  GraphStatistics,
  HealthResponse,
  ResetGraphResponse,
  SearchRequest,
  SearchResponse,
  SubgraphRequest,
} from "./types";


const API_BASE_URL = (
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");


export class ApiError extends Error {
  status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}


interface ValidationError {
  loc?: Array<string | number>;
  msg?: string;
}


function getErrorMessage(detail: unknown): string | null {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const error = detail[0] as ValidationError | undefined;

    if (error?.msg) {
      const field = error.loc?.at(-1);
      return typeof field === "string" ? `${field}: ${error.msg}` : error.msg;
    }
  }

  return null;
}


async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new ApiError(
      "Could not connect to the backend. Check that the API server is running.",
    );
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;

    try {
      const errorBody = (await response.json()) as { detail?: unknown };
      const detailMessage = getErrorMessage(errorBody.detail);

      if (detailMessage) {
        message = detailMessage;
      }
    } catch {
      // Keep the status-based message if the response does not contain JSON.
    }

    throw new ApiError(message, response.status);
  }

  return response.json() as Promise<T>;
}


function postJson<TResponse, TRequest>(
  path: string,
  request: TRequest,
): Promise<TResponse> {
  return apiRequest<TResponse>(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}


export function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/health");
}


export function searchDocuments(
  request: SearchRequest,
): Promise<SearchResponse> {
  return postJson<SearchResponse, SearchRequest>("/search", request);
}


export function adaptiveSearchDocuments(
  request: SearchRequest,
): Promise<AdaptiveSearchResponse> {
  return postJson<AdaptiveSearchResponse, SearchRequest>(
    "/adaptive-search",
    request,
  );
}


export function getGraphStatistics(): Promise<GraphStatistics> {
  return apiRequest<GraphStatistics>("/graph/stats");
}


export function getDocumentDetail(docId: string): Promise<DocumentDetail> {
  return apiRequest<DocumentDetail>(`/documents/${encodeURIComponent(docId)}`);
}


export function resetGraph(): Promise<ResetGraphResponse> {
  return apiRequest<ResetGraphResponse>("/graph/reset", {
    method: "POST",
  });
}


export function getFocusedSubgraph(
  request: SubgraphRequest = {},
): Promise<GraphResponse> {
  return postJson<GraphResponse, SubgraphRequest>("/graph/subgraph", request);
}


export function getFullGraph(): Promise<GraphResponse> {
  return apiRequest<GraphResponse>("/graph");
}
