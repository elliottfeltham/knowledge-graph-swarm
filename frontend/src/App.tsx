import { useEffect, useState } from "react";
import {
  BrainCircuit,
  CircleHelp,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
} from "lucide-react";

import { getHealth } from "./api/client";
import HelpDialog from "./components/HelpDialog";
import SearchPanel from "./components/SearchPanel";

type ConnectionStatus = "checking" | "connected" | "disconnected";

function App() {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("checking");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  async function checkBackend() {
    setConnectionStatus("checking");

    try {
      const health = await getHealth();
      const connected = health.status === "ok" && health.runtime_loaded;

      setConnectionStatus(connected ? "connected" : "disconnected");
    } catch {
      setConnectionStatus("disconnected");
    }
  }

  useEffect(() => {
    void checkBackend();
  }, []);

  const statusLabels: Record<ConnectionStatus, string> = {
    checking: "Checking backend connection",
    connected: "Backend connected",
    disconnected: "Backend unavailable",
  };

  return (
    <div
      className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}
    >
      <aside className="sidebar" aria-label="Search workspace">
        <header className="brand">
          <div className="brand-copy">
            <p className="eyebrow">Adaptive retrieval</p>
            <h1>Knowledge Graph Swarm</h1>
          </div>
          <div className="brand-actions">
            {!sidebarCollapsed && (
              <button
                className="help-button"
                type="button"
                title="About this project"
                aria-label="Open project help"
                onClick={() => setHelpOpen(true)}
              >
                <CircleHelp size={17} aria-hidden="true" />
              </button>
            )}
            <button
              className="sidebar-toggle"
              type="button"
              title={
                sidebarCollapsed
                  ? "Expand search sidebar"
                  : "Collapse search sidebar"
              }
              aria-label={
                sidebarCollapsed
                  ? "Expand search sidebar"
                  : "Collapse search sidebar"
              }
              aria-expanded={!sidebarCollapsed}
              onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            >
              {sidebarCollapsed ? (
                <PanelLeftOpen size={17} aria-hidden="true" />
              ) : (
                <PanelLeftClose size={17} aria-hidden="true" />
              )}
            </button>
          </div>
        </header>

        <div className="sidebar-content">
          <SearchPanel backendAvailable={connectionStatus === "connected"} />
        </div>

        <footer className="runtime-status">
          <span
            className={`status-dot status-dot-${connectionStatus}`}
            aria-hidden="true"
          />
          <span>{statusLabels[connectionStatus]}</span>
          {connectionStatus === "disconnected" && (
            <button
              className="retry-button"
              type="button"
              title="Retry backend connection"
              aria-label="Retry backend connection"
              onClick={() => void checkBackend()}
            >
              <RefreshCw size={14} aria-hidden="true" />
            </button>
          )}
        </footer>
      </aside>

      <main className="graph-workspace" aria-label="Graph workspace">
        <header className="graph-header">
          <div>
            <p className="eyebrow">Graph view</p>
            <h2>Focused network</h2>
          </div>
          <span className="graph-state">Waiting for graph data</span>
        </header>

        <div className="graph-loading" role="status">
          <BrainCircuit size={36} strokeWidth={1.25} aria-hidden="true" />
          <span>Loading workspace</span>
        </div>
      </main>

      {helpOpen && <HelpDialog onClose={() => setHelpOpen(false)} />}
    </div>
  );
}

export default App;
