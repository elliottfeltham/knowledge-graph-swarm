export interface GraphForceSettings {
  centreForce: number;
  repelForce: number;
  linkForce: number;
  linkDistance: number;
}

export const DEFAULT_GRAPH_FORCE_SETTINGS: GraphForceSettings = {
  centreForce: 1,
  repelForce: 30,
  linkForce: 0.2,
  linkDistance: 30,
};

export const FULL_GRAPH_FORCE_SETTINGS: GraphForceSettings = {
  centreForce: 0.5,
  repelForce: 45,
  linkForce: 0.35,
  linkDistance: 40,
};
