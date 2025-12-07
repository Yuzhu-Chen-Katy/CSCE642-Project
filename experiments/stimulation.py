
"""
Run all simulations for the two-step RL project.

Loops over:
    - agent_type: ["mf", "mb", "hybrid"]
    - volatility (drift_std)
    - random seeds

For each condition, uses experiments.trainer.run_training to generate
trial-by-trial logs, and saves one big CSV with columns:

    s1, s2, a1, a2, reward, common, episode,
    agent_type, volatility, seed

Output:
    results/full_results.csv
"""

import os
from typing import List

import pandas as pd

from experiments.trainer import run_training
from utils.config import (
    AGENT_TYPES,
    VOLATILITY_LEVELS,
    N_EPISODES,
    N_SEEDS,
    get_agent_kwargs,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_condition(agent_type: str, sigma: float, seed: int) -> pd.DataFrame:
    """
    Run one (agent_type, volatility, seed) condition.

    Assumes run_training(..., log_behavior=True) returns:
        rewards, log
    where each element of log is a dict with keys like:
        "s1", "s2", "a1", "a2", "reward", "common", "episode"
    """
    env_kwargs = {"drift_std": sigma, "seed": seed}
    agent_kwargs = get_agent_kwargs(agent_type)

    rewards, log = run_training(
        agent_type=agent_type,
        n_episodes=N_EPISODES,
        env_kwargs=env_kwargs,
        agent_kwargs=agent_kwargs,
        verbose=False,
        log_behavior=True,
    )

    # Attach metadata to each row
    records: List[dict] = []
    for row in log:
        a1 = row["first_stage_action"]
        trans = row["transition_type"]

        # encode common/rare as 1/0; None stays None
        if trans == "common":
            common = 1
        elif trans == "rare":
            common = 0
        else:
            common = None

        records.append(
            {
                "episode": int(row["episode"]),
                "a1": a1,
                "reward": float(row["reward"]),
                "common": common,
                "agent_type": agent_type,
                "volatility": sigma,
                "seed": seed,
            }
        )

    return pd.DataFrame(records)


def run_all_simulations() -> pd.DataFrame:
    """Run all (agent, volatility, seed) combos and save a single CSV."""
    all_dfs: List[pd.DataFrame] = []

    for agent_type in AGENT_TYPES:
        for sigma in VOLATILITY_LEVELS:
            for seed in range(N_SEEDS):
                print(f"Simulating agent={agent_type}, sigma={sigma}, seed={seed}...")
                df = simulate_condition(agent_type, sigma, seed)
                all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    out_path = os.path.join(RESULTS_DIR, "full_results.csv")
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")

    return full_df


if __name__ == "__main__":
    run_all_simulations()
