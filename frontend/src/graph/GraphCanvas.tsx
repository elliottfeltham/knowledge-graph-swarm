import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import ForceGraph3D from "react-force-graph-3d";
import type { ForceGraphMethods } from "react-force-graph-3d";

import type {
  GraphEdge,
  GraphNode,
  GraphResponse,
  SearchResult,
} from "../api/types";
import type { GraphForceSettings } from "./graphForceSettings";

interface GraphCanvasProps {
  graph: GraphResponse;
  performanceMode: boolean;
  nodeSelectionEnabled: boolean;
  selectedNodeId: string | null;
  forceSettings: GraphForceSettings;
  searchResults: SearchResult[];
  onNodeSelect: (docId: string) => void;
}

export interface GraphCanvasHandle {
  fitView: () => void;
  resetView: () => void;
}

interface CanvasSize {
  width: number;
  height: number;
}

interface LinkDistanceForce {
  distance: (distance: number) => LinkDistanceForce;
  strength: (strength: number) => LinkDistanceForce;
}

interface StrengthForce {
  strength: (strength: number) => StrengthForce;
}

interface SimulationNode {
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface GraphCanvasNode extends GraphNode, SimulationNode {}

function createOriginForce(strength: number) {
  let nodes: SimulationNode[] = [];

  function originForce(alpha: number) {
    for (const node of nodes) {
      node.vx = (node.vx ?? 0) - (node.x ?? 0) * strength * alpha;
      node.vy = (node.vy ?? 0) - (node.y ?? 0) * strength * alpha;
      node.vz = (node.vz ?? 0) - (node.z ?? 0) * strength * alpha;
    }
  }

  originForce.initialize = (nextNodes: SimulationNode[]) => {
    nodes = nextNodes;
  };

  return originForce;
}

const GraphCanvas = forwardRef<GraphCanvasHandle, GraphCanvasProps>(
  function GraphCanvas(
    {
      graph,
      performanceMode,
      nodeSelectionEnabled,
      selectedNodeId,
      forceSettings,
      searchResults,
      onNodeSelect,
    },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const graphRef = useRef<ForceGraphMethods<GraphCanvasNode, GraphEdge>>(
      undefined,
    );
    const engineReadyRef = useRef(false);
    const appliedSettingsRef = useRef<GraphForceSettings | null>(null);
    const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });

    const graphData = useMemo(
      () => ({
        nodes: graph.nodes.map((node): GraphCanvasNode => ({ ...node })),
        links: graph.edges.map((edge) => ({ ...edge })),
      }),
      [graph],
    );
    const resultRanks = useMemo(
      () =>
        new Map(
          searchResults.map((result, index) => [
            result.doc_id,
            { rank: index + 1, score: result.score },
          ]),
        ),
      [searchResults],
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

    const applyForceSettings = useCallback(
      (settings: GraphForceSettings, reheat: boolean) => {
        if (!graphRef.current) {
          return;
        }

        const linkForce = graphRef.current.d3Force("link") as
          | LinkDistanceForce
          | undefined;
        const centreForce = graphRef.current.d3Force("center") as
          | StrengthForce
          | undefined;
        const repelForce = graphRef.current.d3Force("charge") as
          | StrengthForce
          | undefined;

        centreForce?.strength(settings.centreForce);
        repelForce?.strength(-settings.repelForce);
        linkForce
          ?.strength(settings.linkForce)
          .distance(settings.linkDistance);
        appliedSettingsRef.current = settings;

        if (reheat) {
          graphRef.current.d3ReheatSimulation();
        }
      },
      [],
    );

    useEffect(() => {
      if (
        !engineReadyRef.current ||
        appliedSettingsRef.current === forceSettings
      ) {
        return;
      }

      applyForceSettings(forceSettings, true);
    }, [applyForceSettings, forceSettings]);

    useImperativeHandle(ref, () => ({
      fitView() {
        graphRef.current?.zoomToFit(600, 48);
      },
      resetView() {
        for (const node of graphData.nodes) {
          node.x = (node.x ?? 0) * 0.25;
          node.y = (node.y ?? 0) * 0.25;
          node.z = (node.z ?? 0) * 0.25;
          node.vx = 0;
          node.vy = 0;
          node.vz = 0;
        }

        graphRef.current?.d3ReheatSimulation();
      },
    }));

    return (
      <div
        className="graph-canvas"
        ref={containerRef}
        role="img"
        aria-label={`3D document graph with ${graph.nodes.length} nodes and ${graph.edges.length} edges`}
      >
        {size.width > 0 && size.height > 0 && (
          <ForceGraph3D<GraphCanvasNode, GraphEdge>
            ref={graphRef}
            graphData={graphData}
            width={size.width}
            height={size.height}
            backgroundColor="#101213"
            controlType="orbit"
            enableNavigationControls
            enableNodeDrag={!performanceMode}
            nodeLabel={(node) => {
              const result = resultRanks.get(node.id);
              return result ? `#${result.rank} ${node.title}` : node.title;
            }}
            nodeVal={(node) => {
              const baseSize = Math.max(1, Math.sqrt(node.degree));
              const result = resultRanks.get(node.id);

              if (!result && node.id !== selectedNodeId) {
                return baseSize;
              }

              const rankEmphasis = result
                ? 1 - (result.rank - 1) / searchResults.length
                : 0;
              const selectionEmphasis = node.id === selectedNodeId ? 2 : 0;
              return baseSize + 2 + rankEmphasis * 2 + selectionEmphasis;
            }}
            nodeColor={(node) => {
              if (node.id === selectedNodeId) {
                return "#f3e3a1";
              }

              return resultRanks.has(node.id) ? "#f0b56b" : "#78c9cc";
            }}
            nodeOpacity={0.9}
            nodeResolution={performanceMode ? 5 : 12}
            linkWidth={(edge) => 0.2 + edge.weight * 1.4}
            linkColor={() => "#536a6d"}
            linkOpacity={0.3}
            showNavInfo={false}
            onNodeClick={(node) => {
              if (nodeSelectionEnabled) {
                onNodeSelect(node.id);
              }
            }}
            {...(performanceMode
              ? {
                  cooldownTicks: 120,
                  cooldownTime: 5000,
                  d3AlphaDecay: 0.06,
                }
              : {})}
            onEngineTick={() => {
              if (engineReadyRef.current || !graphRef.current) {
                return;
              }

              engineReadyRef.current = true;
              graphRef.current.d3Force("origin", createOriginForce(0.008));
              applyForceSettings(forceSettings, false);
            }}
            onEngineStop={() => {
              graphRef.current?.zoomToFit(600, 48);
            }}
          />
        )}
      </div>
    );
    },
  );

export default GraphCanvas;
