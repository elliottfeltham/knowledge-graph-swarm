import { fireEvent, render, screen, within } from "@testing-library/react";
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

function renderSearchPanel(
  onSearchComplete = vi.fn(),
  onSearchClear = vi.fn(),
) {
  render(
    <SearchPanel
      backendAvailable
      onSearchStateChange={vi.fn()}
      onSearchComplete={onSearchComplete}
      onSearchClear={onSearchClear}
    />,
  );
}

describe("SearchPanel", () => {
  it("fills the search input from an example question", () => {
    renderSearchPanel();

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
    const handleSearchComplete = vi.fn();
    renderSearchPanel(handleSearchComplete);

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "example query" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Example Document")).toBeInTheDocument();
    expect(screen.getByText("Retrieval time 12.5 ms")).toBeInTheDocument();
    expect(screen.getByText("Retrieval score")).toBeInTheDocument();
    expect(screen.getByText("1 of 10 requested")).toBeInTheDocument();
    expect(
      screen.getByText(/The 3 seeds produced 1 unique candidate/),
    ).toBeInTheDocument();
    expect(searchDocuments).toHaveBeenCalledWith({
      query: "example query",
      top_k: 10,
      seed_k: 3,
      graph_weight: 0.05,
    });
    expect(adaptiveSearchDocuments).not.toHaveBeenCalled();
    expect(handleSearchComplete).toHaveBeenCalledWith(searchResponse);
  });

  it("uses adaptive search when the toggle is enabled", async () => {
    vi.mocked(adaptiveSearchDocuments).mockResolvedValue({
      ...searchResponse,
      adaptive: true,
      update_summary: {
        retrieved_documents: 1,
        strengthened_edges: 3,
        created_edges: 1,
        pruned_edges: 0,
        edge_updates: [],
      },
    });
    renderSearchPanel();

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "adaptive query" },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: "Adaptive search" }));
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Graph updated")).toBeInTheDocument();
    const updateSummary = screen.getByLabelText("Graph update summary");
    expect(within(updateSummary).getByText("Created")).toBeInTheDocument();
    expect(within(updateSummary).getAllByText("1")).toHaveLength(2);
    expect(adaptiveSearchDocuments).toHaveBeenCalledOnce();
    expect(searchDocuments).not.toHaveBeenCalled();
  });

  it("shows API errors", async () => {
    vi.mocked(searchDocuments).mockRejectedValue(
      new Error("The backend could not complete the search."),
    );
    renderSearchPanel();

    fireEvent.change(screen.getByLabelText("Search documents"), {
      target: { value: "example query" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(
      await screen.findByText("The backend could not complete the search."),
    ).toBeInTheDocument();
  });

  it("clears completed search context when the query is cleared", async () => {
    vi.mocked(searchDocuments).mockResolvedValue(searchResponse);
    const handleSearchClear = vi.fn();
    renderSearchPanel(vi.fn(), handleSearchClear);

    const queryInput = screen.getByLabelText("Search documents");
    fireEvent.change(queryInput, { target: { value: "example query" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));

    expect(await screen.findByText("Example Document")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear search" }));

    expect(handleSearchClear).toHaveBeenCalledOnce();
    expect(queryInput).toHaveValue("");
    expect(screen.queryByText("Example Document")).not.toBeInTheDocument();
  });
});
