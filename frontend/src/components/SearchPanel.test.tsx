import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  adaptiveSearchDocuments,
  searchDocuments,
} from "../api/client";
import SearchPanel from "./SearchPanel";

vi.mock("../api/client", () => ({
  searchDocuments: vi.fn(),
  adaptiveSearchDocuments: vi.fn(),
}));

const searchResponse = {
  query: "example query",
  adaptive: false,
  latency_ms: 12.5,
  results: [
    {
      doc_id: "document-1",
      title: "Example Document",
      score: 0.85,
      text_preview: "An example document returned by the search.",
    },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("SearchPanel", () => {
  it("fills the search input from an example question", () => {
    render(<SearchPanel backendAvailable />);

    fireEvent.click(screen.getByText("Example HotpotQA queries"));
    fireEvent.click(
      screen.getByRole("button", {
        name: /Which magazine was started first/,
      }),
    );

    expect(screen.getByLabelText("Search documents")).toHaveValue(
      "Which magazine was started first, Arthur's Magazine or First for Women?",
    );
    expect(searchDocuments).not.toHaveBeenCalled();
  });

  it("runs a read-only search and displays ranked results", async () => {
    vi.mocked(searchDocuments).mockResolvedValue(searchResponse);
    render(<SearchPanel backendAvailable />);

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "example query" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Example Document")).toBeInTheDocument();
    expect(screen.getByText("Retrieval time 12.5 ms")).toBeInTheDocument();
    expect(screen.getByText("Retrieval score")).toBeInTheDocument();
    expect(screen.getByText("1 of 10 requested")).toBeInTheDocument();
    expect(
      screen.getByText(/The 5 seeds produced 1 unique candidate/),
    ).toBeInTheDocument();
    expect(searchDocuments).toHaveBeenCalledWith({
      query: "example query",
      top_k: 10,
      seed_k: 5,
      graph_weight: 0.3,
    });
    expect(adaptiveSearchDocuments).not.toHaveBeenCalled();
  });

  it("uses adaptive search when the toggle is enabled", async () => {
    vi.mocked(adaptiveSearchDocuments).mockResolvedValue({
      ...searchResponse,
      adaptive: true,
      update_summary: {
        retrieved_documents: 1,
        strengthened_edges: 3,
        pruned_edges: 0,
      },
    });
    render(<SearchPanel backendAvailable />);

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "adaptive query" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "Adaptive search" }));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Graph updated")).toBeInTheDocument();
    expect(adaptiveSearchDocuments).toHaveBeenCalledOnce();
    expect(searchDocuments).not.toHaveBeenCalled();
  });

  it("shows API errors", async () => {
    vi.mocked(searchDocuments).mockRejectedValue(
      new Error("The backend could not complete the search."),
    );
    render(<SearchPanel backendAvailable />);

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "example query" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(
      await screen.findByText("The backend could not complete the search."),
    ).toBeInTheDocument();
  });
});
