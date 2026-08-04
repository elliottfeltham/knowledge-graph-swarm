import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getFocusedSubgraph,
  getGraphStatistics,
  resetGraph,
} from "../api/client";
import GraphWorkspace from "./GraphWorkspace";

vi.mock("../api/client", () => ({
  getFocusedSubgraph: vi.fn(),
  getGraphStatistics: vi.fn(),
  resetGraph: vi.fn(),
}));

vi.mock("./GraphCanvas", () => ({
  default: () => <div data-testid="graph-canvas" />,
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getGraphStatistics).mockResolvedValue({
    agents: 100,
    undirected_edges: 350,
    average_degree: 7,
    average_edge_weight: 0.2,
    min_edge_weight: 0.1,
    max_edge_weight: 0.5,
  });
});

function renderGraphWorkspace(
  props: Partial<ComponentProps<typeof GraphWorkspace>> = {},
) {
  render(
    <GraphWorkspace
      backendAvailable
      searchContext={null}
      graphRevision={0}
      {...props}
    />,
  );
}

describe("GraphWorkspace", () => {
  it("waits until the backend is available", () => {
    renderGraphWorkspace({ backendAvailable: false });

    expect(screen.getByText("Waiting for backend connection")).toBeInTheDocument();
    expect(getFocusedSubgraph).not.toHaveBeenCalled();
  });

  it("loads and displays a focused graph", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });

    renderGraphWorkspace();

    expect(await screen.findByTestId("graph-canvas")).toBeInTheDocument();
    expect(screen.getByText("1 nodes / 0 edges")).toBeInTheDocument();
    expect(getFocusedSubgraph).toHaveBeenCalledWith({ max_nodes: 250 });
  });

  it("shows statistics and resets the graph after confirmation", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(resetGraph).mockResolvedValue({
      status: "reset",
      graph_statistics: {
        agents: 100,
        undirected_edges: 350,
        average_degree: 7,
        average_edge_weight: 0.2,
        min_edge_weight: 0.1,
        max_edge_weight: 0.5,
      },
    });

    renderGraphWorkspace();

    fireEvent.click(screen.getByLabelText("Open graph management"));
    expect(await screen.findByText("350")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset graph" }));
    expect(resetGraph).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Confirm reset" }));

    expect(resetGraph).toHaveBeenCalledOnce();
    expect(await screen.findByTestId("graph-canvas")).toBeInTheDocument();
  });

  it("keeps the inspector and graph management panels mutually exclusive", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });

    renderGraphWorkspace();

    expect(await screen.findByLabelText("Graph controls")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Open graph management"));

    await waitFor(() => {
      expect(screen.queryByLabelText("Graph controls")).not.toBeInTheDocument();
    });
    expect(
      screen.queryByRole("button", { name: "Open graph controls" }),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Open graph management"));

    fireEvent.click(
      await screen.findByRole("button", { name: "Open graph controls" }),
    );

    expect(screen.getByLabelText("Graph controls")).toBeInTheDocument();
  });

  it("loads a focused graph around ranked search results", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-2", title: "Document 2", degree: 3 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });

    renderGraphWorkspace({
      searchContext: {
        query: "example query",
        adaptive: false,
        latency_ms: 12,
        results: [
          {
            doc_id: "document-2",
            title: "Document 2",
            score: 0.9,
            text_preview: "Example text",
          },
        ],
      },
    });

    expect(await screen.findByText("Search network")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Clear the search field to return to the default graph view.",
      ),
    ).toBeInTheDocument();
    expect(getFocusedSubgraph).toHaveBeenCalledWith({
      document_ids: ["document-2"],
      depth: 1,
      max_nodes: 250,
    });
  });
});
