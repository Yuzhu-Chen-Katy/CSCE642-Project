# main.py
"""
Clean entry point for graders.

Usage:
    python main.py

This will:
    1) run a small hyperparameter search and save tuned params
    2) run all simulations (agents × volatility × seeds) using tuned params
    3) run full analysis (stay prob, logistic regression, learning curves)
"""

from experiments.hparam_search import run_grid_search
from experiments.stimulation import run_all_simulations
from analysis import run_full_analysis


def main():

    run_grid_search()

    full_df = run_all_simulations()

    run_full_analysis()


if __name__ == "__main__":
    main()
