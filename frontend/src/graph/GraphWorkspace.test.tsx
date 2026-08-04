import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getFocusedSubgraph } from "../api/client";
import GraphWorkspace from "./GraphWorkspace";

vi.mock("../api/client", () => ({
  getFocusedSubgraph: vi.fn(),
}));

vi.mock("./GraphCanvas", () => ({
  default: () => <div data-testid="graph-canvas" />,
}));

beforeEach(() => {
  vi.clearAllMocks();
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
});
