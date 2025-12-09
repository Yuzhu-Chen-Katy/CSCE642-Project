from experiments.hparam_search import run_grid_search
from experiments.stimulation import run_all_simulations
from analysis import run_full_analysis


def main():

    run_grid_search()

    full_df = run_all_simulations()

    run_full_analysis()


if __name__ == "__main__":
    main()
