import { useEffect, useRef, useState } from "react";
import { AlertCircle, BrainCircuit, RefreshCw } from "lucide-react";

import { getFocusedSubgraph } from "../api/client";
import type { GraphResponse } from "../api/types";
import GraphCanvas from "./GraphCanvas";
import type { GraphCanvasHandle } from "./GraphCanvas";
import GraphInspector from "./GraphInspector";
import GraphManagement from "./GraphManagement";
import { DEFAULT_GRAPH_FORCE_SETTINGS } from "./graphForceSettings";
import "./GraphWorkspace.css";

interface GraphWorkspaceProps {
  backendAvailable: boolean;
}

type OpenGraphPanel = "inspector" | "management" | null;

function GraphWorkspace({ backendAvailable }: GraphWorkspaceProps) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forceSettings, setForceSettings] = useState(
    DEFAULT_GRAPH_FORCE_SETTINGS,
  );
  const [openPanel, setOpenPanel] = useState<OpenGraphPanel>("inspector");
  const graphCanvasRef = useRef<GraphCanvasHandle>(null);

  async function loadGraph() {
    if (!backendAvailable) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      setGraph(await getFocusedSubgraph({ max_nodes: 250 }));
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "The graph could not be loaded.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadGraph();
  }, [backendAvailable]);

  const graphState = graph
    ? `${graph.nodes.length} nodes / ${graph.edges.length} edges`
    : loading
      ? "Loading graph"
      : "Waiting for graph data";

  function handlePanelChange(panel: Exclude<OpenGraphPanel, null>, open: boolean) {
    setOpenPanel((currentPanel) => {
      if (open) {
        return panel;
      }

      return currentPanel === panel ? null : currentPanel;
    });
  }

  function resetGraphView() {
    setForceSettings({ ...DEFAULT_GRAPH_FORCE_SETTINGS });
    graphCanvasRef.current?.resetView();
  }

  return (
    <main className="graph-workspace" aria-label="Graph workspace">
      <header className="graph-header">
        <div>
          <p className="eyebrow">Graph view</p>
          <h2>Focused network</h2>
        </div>
        <div className="graph-header-actions">
          <span className="graph-state">{graphState}</span>
          <GraphManagement
            open={openPanel === "management"}
            backendAvailable={backendAvailable}
            onOpenChange={(open) =>
              handlePanelChange("management", open)
            }
            onGraphReset={loadGraph}
          />
        </div>
      </header>

      {graph && graph.nodes.length > 0 && (
        <>
          <GraphCanvas
            ref={graphCanvasRef}
            graph={graph}
            forceSettings={forceSettings}
          />
          {openPanel !== "management" && (
            <GraphInspector
              open={openPanel === "inspector"}
              forceSettings={forceSettings}
              onOpenChange={(open) => handlePanelChange("inspector", open)}
              onForceSettingsChange={setForceSettings}
              onFitView={() => graphCanvasRef.current?.fitView()}
              onResetView={resetGraphView}
            />
          )}
        </>
      )}

      {loading && !graph && (
        <div className="graph-message" role="status">
          <BrainCircuit size={36} strokeWidth={1.25} aria-hidden="true" />
          <span>Loading graph</span>
        </div>
      )}

      {!loading && !graph && !error && (
        <div className="graph-message" role="status">
          <BrainCircuit size={36} strokeWidth={1.25} aria-hidden="true" />
          <span>Waiting for backend connection</span>
        </div>
      )}

      {!loading && graph?.nodes.length === 0 && (
        <div className="graph-message" role="status">
          <BrainCircuit size={36} strokeWidth={1.25} aria-hidden="true" />
          <span>No graph data available</span>
        </div>
      )}

      {error && (
        <div className="graph-message graph-error" role="alert">
          <AlertCircle size={28} aria-hidden="true" />
          <span>{error}</span>
          <button type="button" onClick={() => void loadGraph()}>
            <RefreshCw size={14} aria-hidden="true" />
            Retry
          </button>
        </div>
      )}

      {graph?.truncated && (
        <span className="graph-limit">
          Showing {graph.nodes.length} of {graph.total_nodes} nodes
        </span>
      )}
    </main>
  );
}

export default GraphWorkspace;
