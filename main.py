import logging
import model.simulation
import model.utils
import time
from pprint import pprint

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():

    # ==========================================================
    # Choose mode
    # ==========================================================

    mode = "plot"    # Options:
                        #   "simulate"             - run simulation, save raw_data only
                        #   "analyse"              - load raw_data, compute & save analysis_data
                        #   "simulate_and_analyse" - simulate then analyse without reloading from disk
                        #   "plot"                 - load analysis_data and plot

    # ----------------------------------------------------------
    # simulate_params -- used only in "simulate" and
    # "simulate_and_analyse" modes.
    # ----------------------------------------------------------
    simulate_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 100,
        "time": 1000,
    }

    # ----------------------------------------------------------
    # analyse_params -- used only in "analyse" mode.
    # Identifies which raw_data folders to process.
    # user_number is absent: analysis processes complete run
    # folders, never a subset of users.
    # ----------------------------------------------------------
    analyse_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 1000,
    }

    # ----------------------------------------------------------
    # plot_params -- used only in "plot" mode.
    # Identifies which analysis_data to load.
    # user_number: positive = first N, negative = last N, absent = all
    # ----------------------------------------------------------
    plot_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "time": 1000,
        "user_number": 10,
    }

    plot_list = [
        "mobility",
        #"rg_distribution",
        "freq_distribution",
        #"two_radius",
        #"s_over_time",
        #"rg_over_time",
    ]

    start_time = time.time()
    logger.info(f"Running mode: {mode}")

    # ==========================================================
    # MODE 1: SIMULATE only
    # ==========================================================
    if mode == "simulate":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params)
        simulation.save_raw_data()

    # ==========================================================
    # MODE 2: ANALYSE only
    # ==========================================================
    elif mode == "analyse":

        simulation = model.simulation.Simulation()
        simulation.run_analysis(analyse_params)

    # ==========================================================
    # MODE 3: SIMULATE + ANALYSE (no double disk load)
    # ==========================================================
    elif mode == "simulate_and_analyse":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params)
        raw_run_folder = simulation.save_raw_data()
        simulation.run_analysis(simulate_params, raw_run_folder=raw_run_folder)

    # ==========================================================
    # MODE 4: PLOT only
    # ==========================================================
    elif mode == "plot":

        simulation = model.simulation.Simulation()
        simulation.load_analysis_data(plot_params)
        simulation.plot(plot_list)

    else:
        raise ValueError(
            "Invalid mode. Use: 'simulate', 'analyse', 'simulate_and_analyse', or 'plot'"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()