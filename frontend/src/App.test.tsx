import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

  it("opens and closes the project help", async () => {
    render(<App />);

    const helpButton = screen.getByRole("button", { name: "Open project help" });
    helpButton.focus();
    fireEvent.click(helpButton);

    expect(
      screen.getByRole("dialog", { name: "About this project" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close help" })).toHaveFocus();

    fireEvent.keyDown(window, { key: "Tab" });
    expect(screen.getByRole("button", { name: "Close help" })).toHaveFocus();

    fireEvent.click(screen.getByRole("button", { name: "Close help" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(helpButton).toHaveFocus());
  });

  it("closes the project help with Escape", async () => {
    render(<App />);

    const helpButton = screen.getByRole("button", { name: "Open project help" });
    helpButton.focus();
    fireEvent.click(helpButton);
    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(helpButton).toHaveFocus());
  });
});
