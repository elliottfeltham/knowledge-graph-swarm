import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";
import type { ForceGraphMethods } from "react-force-graph-3d";

import type { GraphEdge, GraphNode, GraphResponse } from "../api/types";

interface GraphCanvasProps {
  graph: GraphResponse;
}

interface CanvasSize {
  width: number;
  height: number;
}

function GraphCanvas({ graph }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphEdge>>(undefined);
  const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });

  const graphData = useMemo(
    () => ({
      nodes: graph.nodes.map((node) => ({ ...node })),
      links: graph.edges.map((edge) => ({ ...edge })),
    }),
    [graph],
  );

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return;
    }

    function updateSize(element: HTMLDivElement) {
      setSize({
        width: element.clientWidth,
        height: element.clientHeight,
      });
    }

    updateSize(container);

    const resizeObserver = new ResizeObserver(() => updateSize(container));
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, []);

  return (
    <div className="graph-canvas" ref={containerRef}>
      {size.width > 0 && size.height > 0 && (
        <ForceGraph3D<GraphNode, GraphEdge>
          ref={graphRef}
          graphData={graphData}
          width={size.width}
          height={size.height}
          backgroundColor="#101213"
          controlType="orbit"
          enableNavigationControls
          enableNodeDrag
          nodeLabel="title"
          nodeVal={(node) => Math.max(1, Math.sqrt(node.degree))}
          nodeColor={() => "#78c9cc"}
          nodeOpacity={0.9}
          nodeResolution={12}
          linkWidth={(edge) => 0.2 + edge.weight * 1.4}
          linkColor={() => "#536a6d"}
          linkOpacity={0.3}
          showNavInfo={false}
          onEngineStop={() => graphRef.current?.zoomToFit(600, 48)}
        />
      )}
    </div>
  );
}

export default GraphCanvas;
