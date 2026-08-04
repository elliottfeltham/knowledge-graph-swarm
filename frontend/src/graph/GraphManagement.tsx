import { useEffect, useState } from "react";
import { AlertCircle, Database, RotateCcw } from "lucide-react";

import { getGraphStatistics, resetGraph } from "../api/client";
import type { GraphStatistics } from "../api/types";

interface GraphManagementProps {
  open: boolean;
  backendAvailable: boolean;
  refreshKey: number;
  onOpenChange: (open: boolean) => void;
  onGraphReset: () => Promise<void>;
}

function GraphManagement({
  open,
  backendAvailable,
  refreshKey,
  onOpenChange,
  onGraphReset,
}: GraphManagementProps) {
  const [statistics, setStatistics] = useState<GraphStatistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!backendAvailable) {
      return;
    }

    async function loadStatistics() {
      setLoading(true);
      setError(null);

      try {
        setStatistics(await getGraphStatistics());
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Graph statistics could not be loaded.",
        );
      } finally {
        setLoading(false);
      }
    }

    void loadStatistics();
  }, [backendAvailable, refreshKey]);

  async function handleReset() {
    setResetting(true);
    setError(null);

    try {
      const response = await resetGraph();
      setStatistics(response.graph_statistics);
      await onGraphReset();
      setConfirmReset(false);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The graph could not be reset.",
      );
    } finally {
      setResetting(false);
    }
  }

  return (
    <details
      className="graph-management"
      open={open}
      onToggle={(event) => onOpenChange(event.currentTarget.open)}
    >
      <summary title="Graph statistics and reset" aria-label="Open graph management">
        <Database size={17} aria-hidden="true" />
      </summary>

      <div className="graph-management-panel">
        <header>
          <div>
            <p className="eyebrow">Graph status</p>
            <h3>Current structure</h3>
          </div>
        </header>

        {loading && !statistics && <p className="management-message">Loading statistics</p>}

        {statistics && (
          <dl className="graph-statistics">
            <div>
              <dt>Documents</dt>
              <dd>{statistics.agents.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Edges</dt>
              <dd>{statistics.undirected_edges.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Average degree</dt>
              <dd>{statistics.average_degree.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Average weight</dt>
              <dd>{statistics.average_edge_weight.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Minimum weight</dt>
              <dd>{statistics.min_edge_weight.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Maximum weight</dt>
              <dd>{statistics.max_edge_weight.toFixed(3)}</dd>
            </div>
          </dl>
        )}

        {error && (
          <p className="management-error" role="alert">
            <AlertCircle size={14} aria-hidden="true" />
            {error}
          </p>
        )}

        <div className="graph-reset">
          {confirmReset ? (
            <div className="reset-confirmation">
              <p>
                Reset the graph to its initial nearest-neighbour structure?
                This removes all adaptive changes.
              </p>
              <div>
                <button
                  type="button"
                  onClick={() => setConfirmReset(false)}
                  disabled={resetting}
                >
                  Cancel
                </button>
                <button
                  className="confirm-reset-button"
                  type="button"
                  onClick={() => void handleReset()}
                  disabled={resetting}
                >
                  {resetting ? "Resetting" : "Confirm reset"}
                </button>
              </div>
            </div>
          ) : (
            <button
              className="reset-graph-button"
              type="button"
              onClick={() => setConfirmReset(true)}
              disabled={!backendAvailable}
            >
              <RotateCcw size={14} aria-hidden="true" />
              Reset graph
            </button>
          )}
        </div>
      </div>
    </details>
  );
}

export default GraphManagement;
