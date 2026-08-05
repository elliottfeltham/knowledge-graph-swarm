import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getFocusedSubgraph,
  getFullGraph,
  getGraphStatistics,
  resetGraph,
} from "../api/client";
import type { GraphResponse } from "../api/types";
import GraphWorkspace from "./GraphWorkspace";

vi.mock("../api/client", () => ({
  getFocusedSubgraph: vi.fn(),
  getFullGraph: vi.fn(),
  getGraphStatistics: vi.fn(),
  resetGraph: vi.fn(),
}));

vi.mock("./GraphCanvas", () => ({
  default: ({
    graph,
    performanceMode,
  }: {
    graph: GraphResponse;
    performanceMode: boolean;
  }) => (
    <div
      data-testid="graph-canvas"
      data-edge-count={graph.edges.length}
      data-performance-mode={performanceMode}
    />
  ),
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

afterEach(() => {
  vi.unstubAllGlobals();
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

  it("starts with graph controls collapsed on narrow screens", async () => {
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({ matches: true }),
    );
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });

    renderGraphWorkspace();

    expect(
      await screen.findByRole("button", { name: "Open graph controls" }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("Graph controls")).not.toBeInTheDocument();
  });

  it("loads the complete graph when full mode is selected", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(getFullGraph).mockResolvedValue({
      nodes: [
        { id: "document-1", title: "Document 1", degree: 2 },
        { id: "document-2", title: "Document 2", degree: 3 },
      ],
      edges: [
        {
          source: "document-1",
          target: "document-2",
          weight: 0.2,
          activation_count: 0,
        },
      ],
      total_nodes: 2,
      total_edges: 1,
      truncated: false,
    });

    renderGraphWorkspace();

    expect(await screen.findByText("Focused network")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Full" }));

    expect(await screen.findByText("Full network")).toBeInTheDocument();
    expect(await screen.findByText("2 nodes / 1 edges")).toBeInTheDocument();
    expect(screen.getByText("Observation mode")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(getFullGraph).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Focused" }));
    await screen.findByText("Focused network");
    fireEvent.click(screen.getByRole("button", { name: "Full" }));

    expect(await screen.findByText("2 nodes / 1 edges")).toBeInTheDocument();
    expect(getFullGraph).toHaveBeenCalledOnce();
  });

  it("limits full mode to the strongest 10,000 visible edges", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 10_001 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(getFullGraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 10_001 }],
      edges: Array.from({ length: 10_001 }, (_, index) => ({
        source: "document-1",
        target: `document-${index + 2}`,
        weight: index / 10_001,
        activation_count: 0,
      })),
      total_nodes: 1,
      total_edges: 10_001,
      truncated: false,
    });

    renderGraphWorkspace();
    fireEvent.click(await screen.findByRole("button", { name: "Full" }));

    const canvas = await screen.findByTestId("graph-canvas");
    expect(canvas).toHaveAttribute("data-edge-count", "10000");
    expect(canvas).toHaveAttribute("data-performance-mode", "true");
    expect(
      screen.getByText("Showing the strongest 10,000 of 10,001 edges"),
    ).toBeInTheDocument();
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

    fireEvent.click(screen.getByLabelText("Close graph management"));

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
      screen.queryByRole("group", { name: "Graph scope" }),
    ).not.toBeInTheDocument();
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
