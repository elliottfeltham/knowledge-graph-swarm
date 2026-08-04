import {
  ChevronRight,
  Maximize2,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";

import "./GraphInspector.css";
import type { GraphForceSettings } from "./graphForceSettings";

interface GraphInspectorProps {
  open: boolean;
  forceSettings: GraphForceSettings;
  onOpenChange: (open: boolean) => void;
  onForceSettingsChange: (settings: GraphForceSettings) => void;
  onFitView: () => void;
  onResetView: () => void;
}

interface ForceControl {
  key: keyof GraphForceSettings;
  label: string;
  min: number;
  max: number;
  step: number;
}

const FORCE_CONTROLS: ForceControl[] = [
  { key: "centreForce", label: "Centre force", min: 0, max: 1, step: 0.1 },
  { key: "repelForce", label: "Repel force", min: 0, max: 200, step: 10 },
  { key: "linkForce", label: "Link force", min: 0, max: 2, step: 0.1 },
  { key: "linkDistance", label: "Link distance", min: 10, max: 100, step: 5 },
];

function GraphInspector({
  open,
  forceSettings,
  onOpenChange,
  onForceSettingsChange,
  onFitView,
  onResetView,
}: GraphInspectorProps) {
  if (!open) {
    return (
      <button
        className="graph-inspector-toggle"
        type="button"
        title="Open graph controls"
        aria-label="Open graph controls"
        onClick={() => onOpenChange(true)}
      >
        <SlidersHorizontal size={17} aria-hidden="true" />
      </button>
    );
  }

  return (
    <aside className="graph-inspector" aria-label="Graph controls">
      <header>
        <div>
          <p className="eyebrow">Graph inspector</p>
          <h3>Controls</h3>
        </div>
        <button
          type="button"
          title="Collapse graph controls"
          aria-label="Collapse graph controls"
          onClick={() => onOpenChange(false)}
        >
          <ChevronRight size={17} aria-hidden="true" />
        </button>
      </header>

      <div className="force-controls">
        {FORCE_CONTROLS.map((control) => {
          const outputId = `${control.key}-value`;

          return (
            <label className="force-control" key={control.key}>
              <span>
                <span>{control.label}</span>
                <output id={outputId}>{forceSettings[control.key]}</output>
              </span>
              <input
                type="range"
                aria-label={control.label}
                aria-describedby={outputId}
                min={control.min}
                max={control.max}
                step={control.step}
                value={forceSettings[control.key]}
                onChange={(event) =>
                  onForceSettingsChange({
                    ...forceSettings,
                    [control.key]: Number(event.target.value),
                  })
                }
              />
            </label>
          );
        })}
      </div>

      <div className="view-actions">
        <button type="button" onClick={onFitView}>
          <Maximize2 size={14} aria-hidden="true" />
          Fit to view
        </button>
        <button type="button" onClick={onResetView}>
          <RotateCcw size={14} aria-hidden="true" />
          Reset view
        </button>
      </div>
    </aside>
  );
}

export default GraphInspector;
