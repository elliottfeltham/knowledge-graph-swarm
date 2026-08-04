import { fireEvent, render, screen } from "@testing-library/react";
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

describe("GraphWorkspace", () => {
  it("waits until the backend is available", () => {
    render(<GraphWorkspace backendAvailable={false} />);

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

    render(<GraphWorkspace backendAvailable />);

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

    render(<GraphWorkspace backendAvailable />);

    fireEvent.click(screen.getByLabelText("Open graph management"));
    expect(await screen.findByText("350")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Reset graph" }));
    expect(resetGraph).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Confirm reset" }));

    expect(resetGraph).toHaveBeenCalledOnce();
    expect(await screen.findByTestId("graph-canvas")).toBeInTheDocument();
  });
});
