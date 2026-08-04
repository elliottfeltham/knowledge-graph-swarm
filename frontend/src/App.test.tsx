import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the application shell", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", { name: "Knowledge Graph Swarm" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Search workspace")).toBeInTheDocument();
    expect(screen.getByLabelText("Graph workspace")).toBeInTheDocument();
  });
});
