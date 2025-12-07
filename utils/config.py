# utils/config.py

AGENT_TYPES = ["mf", "mb", "hybrid"]

AGENT_TYPES_TUNED = ["mf", "mb", "hybrid"]

DEFAULT_SIGMA_TUNE = 0.025

VOLATILITY_LEVELS = [0.015, 0.025, 0.04]

N_EPISODES = 1000
N_SEEDS = 20 # number of random seeds per config

def get_agent_kwargs(agent_type: str):
    if agent_type == "mf":
        return {"alpha": 0.1, "eps": 0.1}
    elif agent_type == "mb":
        return {"alpha": 0.1}
    elif agent_type == "hybrid":
        return {"w": 0.5}
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
