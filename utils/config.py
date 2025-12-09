import os
import json

AGENT_TYPES = ["mf", "mb", "hybrid"]

AGENT_TYPES_TUNED = ["mf", "mb", "hybrid"]

DEFAULT_SIGMA_TUNE = 0.025

VOLATILITY_LEVELS = [0.015, 0.025, 0.04]

N_EPISODES = 1000
N_SEEDS = 20 

RESULTS_DIR = "results"

def get_agent_kwargs(agent_type: str):
    """
    Return hyperparameters for a given agent type.

    If results/best_params.json (written by hparam_search) exists and
    contains tuned parameters for this agent_type, use those.
    Otherwise fall back to fixed defaults.
    """
   
    tuned_path = os.path.join(RESULTS_DIR, "best_params.json")
    if os.path.exists(tuned_path):
        try:
            with open(tuned_path, "r") as f:
                tuned = json.load(f)
            if isinstance(tuned, dict) and agent_type in tuned:
                return tuned[agent_type]
        except Exception as e:
            print(f"Warning: failed to load tuned params from {tuned_path}: {e}")

    if agent_type == "mf":
        return {"alpha": 0.1, "eps": 0.1}
    elif agent_type == "mb":
        return {"alpha": 0.1}
    elif agent_type == "hybrid":
        return {"w": 0.5}
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

