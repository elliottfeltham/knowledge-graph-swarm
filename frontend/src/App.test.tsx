import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./api/client", () => ({
  getHealth: vi.fn().mockResolvedValue({
    status: "ok",
    runtime_loaded: true,
  }),
  searchDocuments: vi.fn(),
  adaptiveSearchDocuments: vi.fn(),
}));

describe("App", () => {
  it("renders the application shell and connects to the backend", async () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: "Knowledge Graph Swarm" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Search workspace")).toBeInTheDocument();
    expect(screen.getByLabelText("Graph workspace")).toBeInTheDocument();
    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
  });

  it("collapses and expands the search sidebar", () => {
    render(<App />);

    fireEvent.click(
      screen.getByRole("button", { name: "Collapse search sidebar" }),
    );

    expect(
      screen.getByRole("button", { name: "Expand search sidebar" }),
    ).toBeInTheDocument();
  });

  it("opens and closes the project help", () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Open project help" }));

    expect(
      screen.getByRole("dialog", { name: "About this project" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close help" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
