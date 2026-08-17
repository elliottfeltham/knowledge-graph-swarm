DEFAULT_SEEDS = [10, 20, 30]

ORIGINAL_CONFIG = {
    "seed_k": 5,
    "graph_weight": 0.3,
    "pruning_enabled": True,
}

OPTIMISED_CONFIG = {
    "seed_k": 3,
    "graph_weight": 0.05,
    "pruning_enabled": False,
    "update_k": 10,
}

SELECTIVE_CONFIG = {
    **OPTIMISED_CONFIG,
    "update_k": 3,
}
