from dataclasses import dataclass


@dataclass
class GraphUpdateConfig:
    strengthen_amount: float = 0.05
    decay_rate: float = 0.995
    prune_threshold: float = 0.05
    max_edge_weight: float = 1.0
