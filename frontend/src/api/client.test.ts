import { afterEach, describe, expect, it, vi } from "vitest";

import {
  adaptiveSearchDocuments,
  ApiError,
  getDocumentDetail,
  getFocusedSubgraph,
  getFullGraph,
  getGraphStatistics,
  getHealth,
  resetGraph,
  searchDocuments,
} from "./client";


function createResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}


afterEach(() => {
  vi.unstubAllGlobals();
});


describe("API client", () => {
  it("requests the backend health status", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      createResponse({ status: "ok", runtime_loaded: true }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await getHealth();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/health",
      undefined,
    );
    expect(response.runtime_loaded).toBe(true);
  });

  it("sends search settings as JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(createResponse({ results: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const request = {
      query: "example query",
      top_k: 10,
      seed_k: 5,
      graph_weight: 0.3,
    };

    await searchDocuments(request);

    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8000/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
  });

  it("maps each graph operation to its backend endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(createResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await adaptiveSearchDocuments({ query: "adaptive query" });
    await getDocumentDetail("document-1");
    await getGraphStatistics();
    await resetGraph();
    await getFocusedSubgraph({ document_ids: ["document-1"] });
    await getFullGraph();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/adaptive-search",
      "http://127.0.0.1:8000/documents/document-1",
      "http://127.0.0.1:8000/graph/stats",
      "http://127.0.0.1:8000/graph/reset",
      "http://127.0.0.1:8000/graph/subgraph",
      "http://127.0.0.1:8000/graph",
    ]);
  });

  it("uses the backend detail when a request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        createResponse({ detail: "Unknown document ID" }, 422),
      ),
    );

    await expect(
      getFocusedSubgraph({ document_ids: ["missing"] }),
    ).rejects.toEqual(new ApiError("Unknown document ID", 422));
  });

  it("turns validation details into a readable message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        createResponse(
          {
            detail: [
              {
                loc: ["body", "top_k"],
                msg: "Input should be less than or equal to 100",
              },
            ],
          },
          422,
        ),
      ),
    );

    await expect(searchDocuments({ query: "example query" })).rejects.toEqual(
      new ApiError("top_k: Input should be less than or equal to 100", 422),
    );
  });

  it("returns a useful message when the backend cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed")));

    await expect(getHealth()).rejects.toThrow(
      "Could not connect to the backend. Check that the API server is running.",
    );
  });
});
