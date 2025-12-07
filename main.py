# main.py
"""
Clean entry point for graders.

Usage:
    python main.py

This will:
    1) run all simulations (agents × volatility × seeds)
    2) run full analysis (stay prob, logistic regression, learning curves)
"""

from experiments.stimulation import run_all_simulations
from analysis import run_full_analysis


def main():
    full_df = run_all_simulations()
    run_full_analysis()


if __name__ == "__main__":
    main()
