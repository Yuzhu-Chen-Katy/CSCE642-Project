
"""
Analysis for the two-step RL project.

Reads:
    results/full_results.csv

Computes:
    - stay (whether first-stage action repeats)
    - prev_reward_bin
    - prev_common_bin
Performs:
    - stay probability analysis
    - logistic regression
    - learning curves

Outputs:
    results/regression_data.csv
    results/stay_summary.csv
    results/logistic_coefs.csv
    results/learning_curves.csv

    figures/stayprob_<agent>.png
    figures/interaction_vs_volatility.png
    figures/learningcurves_<agent>.png
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------- helper transforms ----------

def add_prev_trial_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    From full_results.csv structure:

        s1, s2, a1, a2, reward, common, episode,
        agent_type, volatility, seed

    build:
        prev_a1, prev_reward, prev_common,
        stay, prev_reward_bin, prev_common_bin
    """
    df = df.sort_values(["agent_type", "volatility", "seed", "episode"]).copy()

    group_cols = ["agent_type", "volatility", "seed"]
    df["prev_a1"] = df.groupby(group_cols)["a1"].shift(1)
    df["prev_reward"] = df.groupby(group_cols)["reward"].shift(1)
    df["prev_common"] = df.groupby(group_cols)["common"].shift(1)

    # drop first episode of each run (no previous trial)
    df = df.dropna(subset=["prev_a1"]).copy()

    df["stay"] = (df["a1"] == df["prev_a1"]).astype(int)
    df["prev_reward_bin"] = (df["prev_reward"] > 0).astype(int)
    df["prev_common_bin"] = (df["prev_common"] == 1).astype(int)

    return df


def compute_stay_prob(df_reg: pd.DataFrame) -> pd.DataFrame:
    """Mean stay probability by agent, vol, prev_reward_bin, prev_common_bin."""
    grp = df_reg.groupby(
        ["agent_type", "volatility", "prev_reward_bin", "prev_common_bin"]
    )["stay"]
    summary = grp.agg(["mean", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "stay_prob", "count": "n"})
    return summary


def plot_stay_prob(summary: pd.DataFrame) -> None:
    """Simple stay-prob plot for each agent."""
    agents = summary["agent_type"].unique()
    for agent in agents:
        df_a = summary[summary["agent_type"] == agent]

        plt.figure()
        for common_val, label in [(1, "common"), (0, "rare")]:
            sub = df_a[df_a["prev_common_bin"] == common_val]
            means = (
                sub.groupby("prev_reward_bin")["stay_prob"]
                .mean()
                .reindex([0, 1])
            )
            plt.plot([0, 1], means.values, marker="o", label=label)

        plt.xticks([0, 1], ["no reward", "reward"])
        plt.ylim(0, 1)
        plt.xlabel("Previous reward")
        plt.ylabel("Stay probability")
        plt.title(f"Stay probability vs previous outcome ({agent})")
        plt.legend()
        out_path = os.path.join(FIG_DIR, f"stayprob_{agent}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def fit_logistic_models(df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    For each (agent_type, volatility), fit:

        stay ~ prev_reward_bin + prev_common_bin
               + prev_reward_bin:prev_common_bin
    """
    rows = []
    for (agent_type, vol), sub in df_reg.groupby(["agent_type", "volatility"]):
        # avoid degenerate groups
        if sub["stay"].nunique() < 2:
            continue

        model = smf.logit(
            "stay ~ prev_reward_bin + prev_common_bin + prev_reward_bin:prev_common_bin",
            data=sub,
        ).fit(disp=False)

        params = model.params
        rows.append(
            {
                "agent_type": agent_type,
                "volatility": vol,
                "beta_reward": params.get("prev_reward_bin", np.nan),
                "beta_common": params.get("prev_common_bin", np.nan),
                "beta_interaction": params.get(
                    "prev_reward_bin:prev_common_bin", np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_interaction_vs_volatility(coef_df: pd.DataFrame) -> None:
    """Plot interaction coefficient vs volatility per agent."""
    plt.figure()
    for agent_type in coef_df["agent_type"].unique():
        sub = coef_df[coef_df["agent_type"] == agent_type].sort_values("volatility")
        plt.plot(sub["volatility"], sub["beta_interaction"], marker="o", label=agent_type)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Volatility (drift_std)")
    plt.ylabel("Interaction coefficient (reward × common)")
    plt.title("Logistic regression interaction vs volatility")
    plt.legend()
    out_path = os.path.join(FIG_DIR, "interaction_vs_volatility.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def compute_learning_curves(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean reward per episode across seeds
    for each (agent_type, volatility).
    """
    grp = full_df.groupby(["agent_type", "volatility", "episode"])["reward"]
    curve_df = grp.mean().reset_index().rename(columns={"reward": "mean_reward"})
    return curve_df


def plot_learning_curves(curve_df: pd.DataFrame) -> None:
    """Learning curves by agent + volatility."""
    for agent_type in curve_df["agent_type"].unique():
        plt.figure()
        df_a = curve_df[curve_df["agent_type"] == agent_type]
        for vol in sorted(df_a["volatility"].unique()):
            sub = df_a[df_a["volatility"] == vol]
            plt.plot(sub["episode"], sub["mean_reward"], label=f"sigma={vol}")

        plt.xlabel("Episode")
        plt.ylabel("Mean reward")
        plt.title(f"Learning curves ({agent_type})")
        plt.legend()
        out_path = os.path.join(FIG_DIR, f"learningcurves_{agent_type}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def run_full_analysis() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-level analysis function."""
    full_path = os.path.join(RESULTS_DIR, "full_results.csv")
    if not os.path.exists(full_path):
        raise SystemExit(
            f"Could not find {full_path}. Run experiments/main_experiment.py first."
        )

    full_df = pd.read_csv(full_path)

    # 1) regression-ready data
    reg_df = add_prev_trial_info(full_df)
    reg_path = os.path.join(RESULTS_DIR, "regression_data.csv")
    reg_df.to_csv(reg_path, index=False)

    # 2) stay probabilities
    stay_summary = compute_stay_prob(reg_df)
    stay_path = os.path.join(RESULTS_DIR, "stay_summary.csv")
    stay_summary.to_csv(stay_path, index=False)
    plot_stay_prob(stay_summary)

    # 3) logistic regression
    coef_df = fit_logistic_models(reg_df)
    coef_path = os.path.join(RESULTS_DIR, "logistic_coefs.csv")
    coef_df.to_csv(coef_path, index=False)
    plot_interaction_vs_volatility(coef_df)

    # 4) learning curves
    curve_df = compute_learning_curves(full_df)
    curve_path = os.path.join(RESULTS_DIR, "learning_curves.csv")
    curve_df.to_csv(curve_path, index=False)
    plot_learning_curves(curve_df)

    return reg_df, stay_summary, coef_df


if __name__ == "__main__":
    run_full_analysis()
