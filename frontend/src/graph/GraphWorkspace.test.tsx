import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getDocumentDetail,
  getFocusedSubgraph,
  getFullGraph,
  getGraphStatistics,
  resetGraph,
} from "../api/client";
import type { GraphResponse } from "../api/types";
import GraphWorkspace from "./GraphWorkspace";

vi.mock("../api/client", () => ({
  getDocumentDetail: vi.fn(),
  getFocusedSubgraph: vi.fn(),
  getFullGraph: vi.fn(),
  getGraphStatistics: vi.fn(),
  resetGraph: vi.fn(),
}));

vi.mock("./GraphCanvas", () => ({
  default: ({
    graph,
    performanceMode,
    nodeSelectionEnabled,
    selectedNodeId,
    animationActive,
    highlightActive,
    edgeUpdates,
    onNodeSelect,
  }: {
    graph: GraphResponse;
    performanceMode: boolean;
    nodeSelectionEnabled: boolean;
    selectedNodeId: string | null;
    onNodeSelect: (docId: string) => void;
    animationActive: boolean;
    highlightActive: boolean;
    edgeUpdates: unknown[];
  }) => (
    <div>
      <div
        data-testid="graph-canvas"
        data-edge-count={graph.edges.length}
        data-performance-mode={performanceMode}
        data-selection-enabled={nodeSelectionEnabled}
        data-selected-node={selectedNodeId ?? ""}
        data-animation-active={animationActive}
        data-highlight-active={highlightActive}
        data-edge-updates={edgeUpdates.length}
      />
      {nodeSelectionEnabled && graph.nodes[0] && (
        <button type="button" onClick={() => onNodeSelect(graph.nodes[0].id)}>
          Select first node
        </button>
      )}
    </div>
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
      animationKey={0}
      searchPending={false}
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

  it("shows details when a document node is selected", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(getDocumentDetail).mockResolvedValue({
      doc_id: "document-1",
      title: "Document 1",
      text_preview: "Document text preview",
      degree: 2,
    });

    renderGraphWorkspace();
    fireEvent.click(await screen.findByRole("button", { name: "Select first node" }));

    expect(await screen.findByLabelText("Document details")).toBeInTheDocument();
    expect(screen.getByText("Document text preview")).toBeInTheDocument();
    expect(screen.getByText("document-1")).toBeInTheDocument();
    expect(getDocumentDetail).toHaveBeenCalledWith("document-1");
    expect(screen.getByTestId("graph-canvas")).toHaveAttribute(
      "data-selected-node",
      "document-1",
    );

    fireEvent.click(screen.getByRole("button", { name: "Back to graph controls" }));

    expect(screen.getByLabelText("Graph controls")).toBeInTheDocument();
    expect(screen.getByTestId("graph-canvas")).toHaveAttribute(
      "data-selected-node",
      "",
    );
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
    expect(screen.getByTestId("graph-canvas")).toHaveAttribute(
      "data-selection-enabled",
      "true",
    );
    expect(getFullGraph).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Focused" }));
    await screen.findByText("Focused network");
    fireEvent.click(screen.getByRole("button", { name: "Full" }));

    expect(await screen.findByText("2 nodes / 1 edges")).toBeInTheDocument();
    expect(getFullGraph).toHaveBeenCalledOnce();
  });

  it("shows every edge in full mode", async () => {
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
    expect(canvas).toHaveAttribute("data-edge-count", "10001");
    expect(canvas).toHaveAttribute("data-performance-mode", "true");
  });

  it("opens document details from a full graph node", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(getFullGraph).mockResolvedValue({
      nodes: [{ id: "document-1", title: "Document 1", degree: 2 }],
      edges: [],
      total_nodes: 1,
      total_edges: 0,
      truncated: false,
    });
    vi.mocked(getDocumentDetail).mockResolvedValue({
      doc_id: "document-1",
      title: "Document 1",
      text_preview: "Full graph document preview",
      degree: 2,
    });

    renderGraphWorkspace();
    fireEvent.click(await screen.findByRole("button", { name: "Full" }));
    await screen.findByText("Full network");
    fireEvent.click(await screen.findByRole("button", { name: "Select first node" }));

    expect(await screen.findByLabelText("Document details")).toHaveTextContent(
      "Full graph document preview",
    );
    expect(getDocumentDetail).toHaveBeenCalledWith("document-1");
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
    expect(await screen.findByLabelText("Graph animation legend")).toHaveTextContent(
      "Retrieved document",
    );
    expect(screen.getByTestId("graph-canvas")).toHaveAttribute(
      "data-animation-active",
      "true",
    );
  });

  it("passes adaptive edge changes to the graph animation", async () => {
    vi.mocked(getFocusedSubgraph).mockResolvedValue({
      nodes: [
        { id: "document-1", title: "Document 1", degree: 1 },
        { id: "document-2", title: "Document 2", degree: 1 },
      ],
      edges: [],
      total_nodes: 2,
      total_edges: 0,
      truncated: false,
    });

    renderGraphWorkspace({
      animationKey: 2,
      searchContext: {
        query: "adaptive query",
        adaptive: true,
        latency_ms: 14,
        results: [
          {
            doc_id: "document-1",
            title: "Document 1",
            score: 0.9,
            text_preview: "Example text",
          },
          {
            doc_id: "document-2",
            title: "Document 2",
            score: 0.8,
            text_preview: "Example text",
          },
        ],
        update_summary: {
          retrieved_documents: 2,
          strengthened_edges: 1,
          created_edges: 1,
          pruned_edges: 0,
          edge_updates: [
            {
              source: "document-1",
              target: "document-2",
              previous_weight: 0,
              new_weight: 0.05,
              change_type: "created",
            },
          ],
        },
      },
    });

    expect(await screen.findByTestId("graph-canvas")).toHaveAttribute(
      "data-edge-updates",
      "1",
    );
    expect(screen.getByLabelText("Graph animation legend")).toHaveTextContent(
      "StrengthenedCreatedPruned",
    );
  });
});
