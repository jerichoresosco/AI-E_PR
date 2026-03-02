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
    # 1) Choose mode
    # ==========================================================

    mode = "analyse"   # Options:
                                    #   "simulate"             – run simulation, save raw_data only
                                    #   "analyse"              – load raw_data, compute & save analysis_data
                                    #   "simulate_and_analyse" – simulate then analyse without reloading from disk
                                    #   "plot"                 – load analysis_data and plot

    # Parameters for simulation / folder matching
    simulation_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 100,
        "time": 1000
    }

    # Filters for loading saved datasets (None = ignore that filter)
    filters = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.75,
        "b_param": 0.6,
        "user_number": 10,   # positive → first N, negative → last N, absent → all
        "time": 1000
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
    #   Runs the simulation and saves user_history + venue_history
    #   to raw_data/. No statistics are computed.
    # ==========================================================
    if mode == "simulate":

        simulation = model.simulation.Simulation(
            **simulation_params,
            #plots_to_generate=plot_list,
            #sim_params=simulation_params
        )
        simulation.run()
        simulation.save_raw_data()

    # ==========================================================
    # MODE 2: ANALYSE only
    #   Loads raw_data via filters, computes analysis quantities
    #   (stats, rg_over_time, s_over_time, venue copy) and saves
    #   them to analysis_data/.
    # ==========================================================
    elif mode == "analyse":

        simulation = model.simulation.Simulation(
            **simulation_params,
            #plots_to_generate=plot_list,
            #sim_params=simulation_params,
            filters=filters
        )
        # raw_run_folder=None → loads from disk using filters
        simulation.run_analysis(raw_run_folder=None)

    # ==========================================================
    # MODE 3: SIMULATE + ANALYSE (no double disk load)
    #   Runs simulation, saves raw_data, then immediately runs
    #   analysis on the in-memory users_list (no reload needed).
    # ==========================================================
    elif mode == "simulate_and_analyse":

        simulation = model.simulation.Simulation(
            **simulation_params,
            #plots_to_generate=plot_list,
            #sim_params=simulation_params
        )
        simulation.run()
        raw_run_folder = simulation.save_raw_data()

        # Pass raw_run_folder so analysis skips the disk reload
        simulation.run_analysis(raw_run_folder=raw_run_folder)

    # ==========================================================
    # MODE 4: PLOT only
    #   Loads pre-computed analysis_data via filters and plots.
    # ==========================================================
    elif mode == "plot":

        simulation = model.simulation.Simulation(
            **simulation_params,
            plots_to_generate=plot_list,
            sim_params=simulation_params,
            filters=filters
        )
        simulation.load_analysis_data()
        simulation.plot()

    else:
        raise ValueError(
            "Invalid mode. Use: 'simulate', 'analyse', 'simulate_and_analyse', or 'plot'"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()