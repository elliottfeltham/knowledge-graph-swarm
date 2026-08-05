import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, BrainCircuit, RefreshCw } from "lucide-react";

import {
  getDocumentDetail,
  getFocusedSubgraph,
  getFullGraph,
} from "../api/client";
import type {
  DocumentDetail,
  GraphResponse,
  SearchResponse,
} from "../api/types";
import DocumentInspector from "./DocumentInspector";
import GraphCanvas from "./GraphCanvas";
import type { GraphCanvasHandle } from "./GraphCanvas";
import GraphInspector from "./GraphInspector";
import GraphManagement from "./GraphManagement";
import { DEFAULT_GRAPH_FORCE_SETTINGS } from "./graphForceSettings";
import "./GraphWorkspace.css";

interface GraphWorkspaceProps {
  backendAvailable: boolean;
  searchContext: SearchResponse | null;
  graphRevision: number;
}

type OpenGraphPanel = "inspector" | "management" | null;
type GraphViewMode = "focused" | "full";

const FULL_GRAPH_EDGE_LIMIT = 10_000;

function getInitialOpenPanel(): OpenGraphPanel {
  if (window.matchMedia?.("(max-width: 760px)").matches) {
    return null;
  }

  return "inspector";
}

function GraphWorkspace({
  backendAvailable,
  searchContext,
  graphRevision,
}: GraphWorkspaceProps) {
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forceSettings, setForceSettings] = useState(
    DEFAULT_GRAPH_FORCE_SETTINGS,
  );
  const [viewMode, setViewMode] = useState<GraphViewMode>("focused");
  const [openPanel, setOpenPanel] = useState<OpenGraphPanel>(getInitialOpenPanel);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] =
    useState<DocumentDetail | null>(null);
  const [documentLoading, setDocumentLoading] = useState(false);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const graphCanvasRef = useRef<GraphCanvasHandle>(null);
  const graphRequestRef = useRef(0);
  const documentRequestRef = useRef(0);
  const fullGraphCacheRef = useRef<{
    revision: number;
    graph: GraphResponse;
  } | null>(null);

  const loadGraph = useCallback(async () => {
    if (!backendAvailable) {
      return;
    }

    const requestNumber = graphRequestRef.current + 1;
    graphRequestRef.current = requestNumber;
    documentRequestRef.current += 1;
    setLoading(true);
    setGraph(null);
    setError(null);
    setSelectedNodeId(null);
    setSelectedDocument(null);
    setDocumentLoading(false);
    setDocumentError(null);

    try {
      const documentIds = searchContext?.results.map((result) => result.doc_id);

      if (searchContext && !documentIds?.length) {
        setGraph({
          nodes: [],
          edges: [],
          total_nodes: 0,
          total_edges: 0,
          truncated: false,
        });
        setLoading(false);
        return;
      }

      let nextGraph: GraphResponse;

      if (!searchContext && viewMode === "full") {
        const cachedGraph = fullGraphCacheRef.current;

        if (cachedGraph?.revision === graphRevision) {
          nextGraph = cachedGraph.graph;
        } else {
          nextGraph = await getFullGraph();
          fullGraphCacheRef.current = {
            revision: graphRevision,
            graph: nextGraph,
          };
        }
      } else {
        nextGraph = await getFocusedSubgraph({
          ...(documentIds?.length
            ? { document_ids: documentIds, depth: 1 }
            : {}),
          max_nodes: 250,
        });
      }

      if (graphRequestRef.current === requestNumber) {
        setGraph(nextGraph);
      }
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "The graph could not be loaded.";

      if (graphRequestRef.current === requestNumber) {
        setError(message);
      }
    } finally {
      if (graphRequestRef.current === requestNumber) {
        setLoading(false);
      }
    }
  }, [backendAvailable, graphRevision, searchContext, viewMode]);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  const fullGraphActive = !searchContext && viewMode === "full";
  const displayedGraph = useMemo(() => {
    if (!graph || !fullGraphActive || graph.edges.length <= FULL_GRAPH_EDGE_LIMIT) {
      return graph;
    }

    return {
      ...graph,
      edges: [...graph.edges]
        .sort((first, second) => second.weight - first.weight)
        .slice(0, FULL_GRAPH_EDGE_LIMIT),
    };
  }, [fullGraphActive, graph]);
  const edgesLimited = Boolean(
    displayedGraph && graph && displayedGraph.edges.length < graph.edges.length,
  );

  const graphState = loading
    ? viewMode === "full" && !searchContext
      ? "Loading full graph"
      : "Loading graph"
    : displayedGraph
      ? `${displayedGraph.nodes.length} nodes / ${displayedGraph.edges.length} edges`
      : "Waiting for graph data";

  const graphTitle = searchContext
    ? "Search network"
    : viewMode === "full"
      ? "Full network"
      : "Focused network";

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

  async function selectDocument(docId: string) {
    if (fullGraphActive) {
      return;
    }

    const requestNumber = documentRequestRef.current + 1;
    documentRequestRef.current = requestNumber;
    setSelectedNodeId(docId);
    setSelectedDocument(null);
    setDocumentLoading(true);
    setDocumentError(null);
    setOpenPanel("inspector");

    try {
      const document = await getDocumentDetail(docId);

      if (documentRequestRef.current === requestNumber) {
        setSelectedDocument(document);
      }
    } catch (requestError) {
      if (documentRequestRef.current === requestNumber) {
        setDocumentError(
          requestError instanceof Error
            ? requestError.message
            : "The document could not be loaded.",
        );
      }
    } finally {
      if (documentRequestRef.current === requestNumber) {
        setDocumentLoading(false);
      }
    }
  }

  function clearDocumentSelection() {
    documentRequestRef.current += 1;
    setSelectedNodeId(null);
    setSelectedDocument(null);
    setDocumentLoading(false);
    setDocumentError(null);
  }

  async function handleGraphReset() {
    fullGraphCacheRef.current = null;
    await loadGraph();
  }

  const selectedResultIndex = searchContext?.results.findIndex(
    (result) => result.doc_id === selectedNodeId,
  ) ?? -1;
  const selectedSearchResult =
    selectedResultIndex >= 0 && searchContext
      ? searchContext.results[selectedResultIndex]
      : null;

  return (
    <main className="graph-workspace" aria-label="Graph workspace">
      <header className="graph-header">
        <div>
          <p className="eyebrow">Graph view</p>
          <h2>{graphTitle}</h2>
        </div>
        <div className="graph-header-actions">
          {!searchContext && (
            <div
              className="graph-view-switch"
              role="group"
              aria-label="Graph scope"
            >
              <button
                type="button"
                aria-pressed={viewMode === "focused"}
                onClick={() => setViewMode("focused")}
              >
                Focused
              </button>
              <button
                type="button"
                aria-pressed={viewMode === "full"}
                onClick={() => setViewMode("full")}
              >
                Full
              </button>
            </div>
          )}
          <span className="graph-state" aria-live="polite">
            {graphState}
          </span>
          <GraphManagement
            open={openPanel === "management"}
            backendAvailable={backendAvailable}
            refreshKey={graphRevision}
            onOpenChange={(open) =>
              handlePanelChange("management", open)
            }
            onGraphReset={handleGraphReset}
          />
        </div>
      </header>

      {displayedGraph && displayedGraph.nodes.length > 0 && (
        <>
          <GraphCanvas
            ref={graphCanvasRef}
            graph={displayedGraph}
            performanceMode={fullGraphActive}
            nodeSelectionEnabled={!fullGraphActive}
            selectedNodeId={selectedNodeId}
            forceSettings={forceSettings}
            searchResults={searchContext?.results ?? []}
            onNodeSelect={(docId) => void selectDocument(docId)}
          />
          {openPanel !== "management" && (
            selectedNodeId ? (
              <DocumentInspector
                document={selectedDocument}
                loading={documentLoading}
                error={documentError}
                searchResult={selectedSearchResult}
                searchRank={
                  selectedResultIndex >= 0 ? selectedResultIndex + 1 : null
                }
                onBack={clearDocumentSelection}
                onRetry={() => void selectDocument(selectedNodeId)}
              />
            ) : (
              <GraphInspector
                open={openPanel === "inspector"}
                observationMode={fullGraphActive}
                searchNetworkActive={searchContext !== null}
                forceSettings={forceSettings}
                onOpenChange={(open) => handlePanelChange("inspector", open)}
                onForceSettingsChange={setForceSettings}
                onFitView={() => graphCanvasRef.current?.fitView()}
                onResetView={resetGraphView}
              />
            )
          )}
        </>
      )}

      {loading && !graph && (
        <div className="graph-message" role="status">
          <BrainCircuit size={36} strokeWidth={1.25} aria-hidden="true" />
          <span>
            {viewMode === "full" && !searchContext
              ? "Loading full graph. This may take a moment."
              : "Loading graph"}
          </span>
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

      {edgesLimited && graph && displayedGraph && (
        <span className="graph-limit">
          Showing the strongest {displayedGraph.edges.length.toLocaleString()} of{" "}
          {graph.edges.length.toLocaleString()} edges
        </span>
      )}
    </main>
  );
}

export default GraphWorkspace;
