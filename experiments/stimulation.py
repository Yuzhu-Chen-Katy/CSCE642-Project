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

    records: List[dict] = []
    for row in log:
        a1 = row["first_stage_action"]
        trans = row["transition_type"]

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
