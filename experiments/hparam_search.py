# experiments/hparam_search.py

import os
from typing import List, Dict

import numpy as np
import pandas as pd

from experiments.trainer import run_training
from utils.config import DEFAULT_SIGMA_TUNE

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Grids we will search over:
ALPHAS = [0.05, 0.1, 0.2]
EPSILONS = [0.05, 0.1, 0.2]
WS = [0.25, 0.5, 0.75]

N_EPISODES_TUNE = 300
N_SEEDS_TUNE = 5


def evaluate_setting(agent_type: str, agent_kwargs: Dict) -> float:
    """
    Run N_SEEDS_TUNE simulations for a given (agent_type, agent_kwargs)
    in the medium-volatility condition and return the average reward over
    the last third of episodes.
    """
    seed_rewards: List[float] = []

    for seed in range(N_SEEDS_TUNE):
        env_kwargs = {"drift_std": DEFAULT_SIGMA_TUNE, "seed": seed}

        result = run_training(
            agent_type=agent_type,
            n_episodes=N_EPISODES_TUNE,
            env_kwargs=env_kwargs,
            agent_kwargs=agent_kwargs,
            verbose=False,
            log_behavior=False,
        )

        # Accept both rewards-only and (rewards, ...) return formats
        if isinstance(result, tuple):
            rewards = result[0]
        else:
            rewards = result

        rewards = np.asarray(rewards, dtype=float)
        last_third = rewards[int(2 * N_EPISODES_TUNE / 3):]
        seed_rewards.append(last_third.mean())

    return float(np.mean(seed_rewards))


def run_grid_search() -> pd.DataFrame:
    rows = []

    # ----- Model-Free agent: tune alpha and eps -----
    for alpha in ALPHAS:
        for eps in EPSILONS:
            agent_type = "mf"
            agent_kwargs = {"alpha": alpha, "eps": eps}
            print(f"Tuning MF: alpha={alpha}, eps={eps}")
            score = evaluate_setting(agent_type, agent_kwargs)
            rows.append(
                {
                    "agent_type": agent_type,
                    "alpha": alpha,
                    "eps_or_epsilon_or_w": eps,
                    "mean_reward_last_third": score,
                }
            )

    # ----- Model-Based agent: tune alpha and epsilon -----
    for alpha in ALPHAS:
        for epsilon in EPSILONS:
            agent_type = "mb"
            # MBAgent(alpha=..., beta=5.0 (default), epsilon=...)
            agent_kwargs = {"alpha": alpha, "epsilon": epsilon}
            print(f"Tuning MB: alpha={alpha}, epsilon={epsilon}")
            score = evaluate_setting(agent_type, agent_kwargs)
            rows.append(
                {
                    "agent_type": agent_type,
                    "alpha": alpha,
                    "eps_or_epsilon_or_w": epsilon,
                    "mean_reward_last_third": score,
                }
            )

    # ----- Hybrid agent: tune mixing weight w -----
    for w in WS:
        agent_type = "hybrid"
        agent_kwargs = {"w": w}
        print(f"Tuning Hybrid: w={w}")
        score = evaluate_setting(agent_type, agent_kwargs)
        rows.append(
            {
                "agent_type": agent_type,
                "alpha": None,
                "eps_or_epsilon_or_w": w,
                "mean_reward_last_third": score,
            }
        )

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "tuning_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved tuning results to {out_path}")
    return df


if __name__ == "__main__":
    run_grid_search()
