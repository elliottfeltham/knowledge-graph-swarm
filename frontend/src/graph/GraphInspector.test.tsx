import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import GraphInspector from "./GraphInspector";
import { DEFAULT_GRAPH_FORCE_SETTINGS } from "./graphForceSettings";

describe("GraphInspector", () => {
  it("updates graph controls", () => {
    const handleSettingsChange = vi.fn();
    const handleFitView = vi.fn();
    const handleResetView = vi.fn();

    render(
      <GraphInspector
        open
        forceSettings={DEFAULT_GRAPH_FORCE_SETTINGS}
        onOpenChange={vi.fn()}
        onForceSettingsChange={handleSettingsChange}
        onFitView={handleFitView}
        onResetView={handleResetView}
      />,
    );

    fireEvent.change(screen.getByRole("slider", { name: "Repel force" }), {
      target: { value: "80" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Fit to view" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset view" }));

    expect(handleSettingsChange).toHaveBeenCalledWith({
      ...DEFAULT_GRAPH_FORCE_SETTINGS,
      repelForce: 80,
    });
    expect(handleFitView).toHaveBeenCalledOnce();
    expect(handleResetView).toHaveBeenCalledOnce();
  });

  it("requests that the controls are collapsed", () => {
    const handleOpenChange = vi.fn();

    render(
      <GraphInspector
        open
        forceSettings={DEFAULT_GRAPH_FORCE_SETTINGS}
        onOpenChange={handleOpenChange}
        onForceSettingsChange={vi.fn()}
        onFitView={vi.fn()}
        onResetView={vi.fn()}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Collapse graph controls" }),
    );

    expect(handleOpenChange).toHaveBeenCalledWith(false);
  });
});
