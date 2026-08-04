import { BrainCircuit } from "lucide-react";

function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Search workspace">
        <header className="brand">
          <div>
            <p className="eyebrow">Adaptive retrieval</p>
            <h1>Knowledge Graph Swarm</h1>
          </div>
        </header>

        <div className="sidebar-content">
          <section className="loading-section" aria-label="Search panel loading">
            <div className="loading-line loading-line-short" />
            <div className="loading-field" />
            <div className="loading-line" />
            <div className="loading-line" />
          </section>
        </div>

        <footer className="runtime-status">
          <span className="status-dot" aria-hidden="true" />
          Backend connection pending
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
    </div>
  );
}

export default App;
