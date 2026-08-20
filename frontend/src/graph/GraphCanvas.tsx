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
  GraphEdgeChangeType,
  GraphEdgeUpdate,
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
  animationKey: number | null;
  animationActive: boolean;
  highlightActive: boolean;
  edgeUpdates: GraphEdgeUpdate[];
  onNodeSelect: (docId: string) => void;
  onInitialLayoutComplete?: () => void;
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
  strength: (
    strength: number | ((edge: AnimatedGraphEdge) => number),
  ) => LinkDistanceForce;
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

interface AnimatedGraphEdge extends GraphEdge {
  animationChangeType?: GraphEdgeChangeType;
  edgeKey: string;
}

const GRAPH_ANIMATION_DURATION_MS = 3_000;
const MAX_PARTICLE_EDGES = 50;
const FULL_GRAPH_FORCE_EDGE_LIMIT = 10_000;
const STRENGTHENED_EDGE_COLOR = "#ffd43b";
const CREATED_EDGE_COLOR = "#00e5ff";
const DECAYED_EDGE_COLOR = "#b388ff";
const PRUNED_EDGE_COLOR = "#ff4d6d";

function makeEdgeKey(source: string, target: string): string {
  return [source, target].sort().join("::");
}

function getEdgeNodeId(node: string | GraphCanvasNode): string {
  return typeof node === "string" ? node : node.id;
}

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
      animationKey,
      animationActive,
      highlightActive,
      edgeUpdates,
      onNodeSelect,
      onInitialLayoutComplete,
    },
    ref,
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const graphRef = useRef<ForceGraphMethods<GraphCanvasNode, AnimatedGraphEdge>>(
      undefined,
    );
    const engineReadyRef = useRef(false);
    const appliedSettingsRef = useRef<GraphForceSettings | null>(null);
    const hasAutomaticallyFittedRef = useRef(false);
    const layoutCompleteTimerRef = useRef<number | null>(null);
    const [size, setSize] = useState<CanvasSize>({ width: 0, height: 0 });
    const [animationProgress, setAnimationProgress] = useState(1);
    const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
    const [reducedMotion, setReducedMotion] = useState(
      () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    );

    const graphData = useMemo(() => {
      const visibleNodeIds = new Set(graph.nodes.map((node) => node.id));
      const updateByEdge = new Map(
        edgeUpdates.map((update) => [
          makeEdgeKey(update.source, update.target),
          update,
        ]),
      );
      const links: AnimatedGraphEdge[] = graph.edges.map((edge) => {
        const edgeKey = makeEdgeKey(edge.source, edge.target);
        return {
          ...edge,
          edgeKey,
          animationChangeType: updateByEdge.get(edgeKey)?.change_type,
        };
      });
      const existingEdgeKeys = new Set(links.map((edge) => edge.edgeKey));

      if (highlightActive) {
        for (const update of edgeUpdates) {
          const edgeKey = makeEdgeKey(update.source, update.target);

          if (
            update.change_type !== "pruned" ||
            existingEdgeKeys.has(edgeKey) ||
            !visibleNodeIds.has(update.source) ||
            !visibleNodeIds.has(update.target)
          ) {
            continue;
          }

          links.push({
            source: update.source,
            target: update.target,
            weight: update.previous_weight,
            edgeKey,
            animationChangeType: "pruned",
          });
        }
      }

      return {
        nodes: graph.nodes.map((node): GraphCanvasNode => ({ ...node })),
        links,
      };
    }, [edgeUpdates, graph, highlightActive]);
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
    const particleEdgeKeys = useMemo(
      () =>
        new Set(
          edgeUpdates
            .filter(
              (update) =>
                update.change_type === "strengthened" ||
                update.change_type === "created",
            )
            .sort(
              (first, second) =>
                Math.abs((second.new_weight ?? 0) - second.previous_weight) -
                Math.abs((first.new_weight ?? 0) - first.previous_weight),
            )
            .slice(0, MAX_PARTICLE_EDGES)
            .map((update) => makeEdgeKey(update.source, update.target)),
        ),
      [edgeUpdates],
    );
    const forceEdgeKeys = useMemo(() => {
      if (!performanceMode || graphData.links.length <= FULL_GRAPH_FORCE_EDGE_LIMIT) {
        return null;
      }

      return new Set(
        [...graphData.links]
          .sort((first, second) => second.weight - first.weight)
          .slice(0, FULL_GRAPH_FORCE_EDGE_LIMIT)
          .map((edge) => edge.edgeKey),
      );
    }, [graphData.links, performanceMode]);

    useEffect(() => {
      const mediaQuery = window.matchMedia?.("(prefers-reduced-motion: reduce)");

      if (!mediaQuery) {
        return;
      }

      const handleChange = () => setReducedMotion(mediaQuery.matches);
      mediaQuery.addEventListener?.("change", handleChange);
      return () => mediaQuery.removeEventListener?.("change", handleChange);
    }, []);

    useEffect(() => {
      if (!animationActive) {
        setAnimationProgress(1);
        return;
      }

      if (reducedMotion) {
        setAnimationProgress(0.35);
        return;
      }

      const startTime = performance.now();
      let frameId = 0;

      const updateAnimation = (currentTime: number) => {
        const progress = Math.min(
          (currentTime - startTime) / GRAPH_ANIMATION_DURATION_MS,
          1,
        );
        setAnimationProgress(progress);

        if (progress < 1) {
          frameId = requestAnimationFrame(updateAnimation);
        }
      };

      frameId = requestAnimationFrame(updateAnimation);
      return () => cancelAnimationFrame(frameId);
    }, [animationActive, animationKey, reducedMotion]);

    useEffect(() => {
      engineReadyRef.current = false;
      appliedSettingsRef.current = null;
      hasAutomaticallyFittedRef.current = false;

      if (layoutCompleteTimerRef.current !== null) {
        window.clearTimeout(layoutCompleteTimerRef.current);
        layoutCompleteTimerRef.current = null;
      }

      return () => {
        if (layoutCompleteTimerRef.current !== null) {
          window.clearTimeout(layoutCompleteTimerRef.current);
        }
      };
    }, [graph]);

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
          ?.strength(
            forceEdgeKeys
              ? (edge) =>
                  forceEdgeKeys.has(edge.edgeKey) ? settings.linkForce : 0
              : settings.linkForce,
          )
          .distance(settings.linkDistance);
        appliedSettingsRef.current = settings;

        if (reheat) {
          graphRef.current.d3ReheatSimulation();
        }
      },
      [forceEdgeKeys],
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
          <ForceGraph3D<GraphCanvasNode, AnimatedGraphEdge>
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
              const hoverEmphasis = node.id === hoveredNodeId ? 1.5 : 0;
              const pulse =
                animationActive && result && !reducedMotion
                  ? (Math.sin(animationProgress * Math.PI * 4) + 1) * 0.75
                  : 0;
              return (
                baseSize +
                2 +
                rankEmphasis * 2 +
                selectionEmphasis +
                hoverEmphasis +
                pulse
              );
            }}
            nodeColor={(node) => {
              if (node.id === selectedNodeId) {
                return "#f3e3a1";
              }

              if (resultRanks.has(node.id)) {
                return "#f0b56b";
              }

              if (animationActive) {
                return "#405456";
              }

              return highlightActive ? "#668f92" : "#78c9cc";
            }}
            nodeOpacity={0.9}
            nodeResolution={12}
            linkWidth={(edge) => {
              const normalWidth = 0.2 + edge.weight * 1.4;
              const connectedToHoveredNode =
                hoveredNodeId !== null &&
                (getEdgeNodeId(edge.source) === hoveredNodeId ||
                  getEdgeNodeId(edge.target) === hoveredNodeId);
              const settleProgress = Math.max(
                0,
                (animationProgress - 0.47) / 0.53,
              );

              if (!highlightActive || !edge.animationChangeType) {
                return connectedToHoveredNode ? normalWidth + 1.2 : normalWidth;
              }

              if (animationActive && edge.animationChangeType === "created") {
                return normalWidth * Math.max(0.15, animationProgress / 0.47);
              }

              if (edge.animationChangeType === "pruned") {
                return animationActive
                  ? Math.max(0.02, 2 * (1 - settleProgress))
                  : 0;
              }

              if (edge.animationChangeType === "decayed") {
                return animationActive
                  ? normalWidth + 0.8 * (1 - settleProgress)
                  : normalWidth + 0.2;
              }

              if (animationActive) {
                return normalWidth + 2.2 * (1 - settleProgress);
              }

              const updateEmphasis =
                edge.animationChangeType === "strengthened" ? 0.7 : 0.4;
              return normalWidth + updateEmphasis;
            }}
            linkColor={(edge) => {
              const connectedToHoveredNode =
                hoveredNodeId !== null &&
                (getEdgeNodeId(edge.source) === hoveredNodeId ||
                  getEdgeNodeId(edge.target) === hoveredNodeId);

              if (
                highlightActive &&
                edge.animationChangeType === "strengthened"
              ) {
                return STRENGTHENED_EDGE_COLOR;
              }

              if (highlightActive && edge.animationChangeType === "created") {
                return CREATED_EDGE_COLOR;
              }

              if (highlightActive && edge.animationChangeType === "decayed") {
                return DECAYED_EDGE_COLOR;
              }

              if (highlightActive && edge.animationChangeType === "pruned") {
                return PRUNED_EDGE_COLOR;
              }

              if (connectedToHoveredNode) {
                return "#a5e4e6";
              }

              if (hoveredNodeId) {
                return "#354244";
              }

              if (highlightActive) {
                return animationActive ? "#354244" : "#485c5e";
              }

              return "#536a6d";
            }}
            linkOpacity={0.3}
            linkDirectionalParticles={(edge) =>
              animationActive &&
              !reducedMotion &&
              particleEdgeKeys.has(edge.edgeKey)
                ? 2
                : 0
            }
            linkDirectionalParticleWidth={1.5}
            linkDirectionalParticleSpeed={0.008}
            linkDirectionalParticleColor={(edge) =>
              edge.animationChangeType === "created"
                ? CREATED_EDGE_COLOR
                : STRENGTHENED_EDGE_COLOR
            }
            showNavInfo={false}
            onNodeHover={(node) => setHoveredNodeId(node?.id ?? null)}
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
              if (!hasAutomaticallyFittedRef.current) {
                hasAutomaticallyFittedRef.current = true;
                graphRef.current?.zoomToFit(600, 48);

                if (onInitialLayoutComplete) {
                  layoutCompleteTimerRef.current = window.setTimeout(
                    onInitialLayoutComplete,
                    650,
                  );
                }
              }
            }}
          />
        )}
      </div>
    );
    },
  );

export default GraphCanvas;
