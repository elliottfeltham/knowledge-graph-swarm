import { BrainCircuit } from "lucide-react";

import "./GraphWorkspace.css";

function GraphWorkspace() {
  return (
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
  );
}

export default GraphWorkspace;
