DEFAULT_SEEDS = [10, 20, 30]

STRENGTHEN_AMOUNT = 0.01
DECAY_RATE = 0.995

ORIGINAL_CONFIG = {
    "seed_k": 5,
    "graph_weight": 0.3,
    "pruning_enabled": True,
}

OPTIMISED_CONFIG = {
    "seed_k": 3,
    "graph_weight": 0.05,
    "pruning_enabled": False,
    "update_k": 3,
    "strengthen_amount": STRENGTHEN_AMOUNT,
    "decay_rate": DECAY_RATE,
}
