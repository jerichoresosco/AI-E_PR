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

    mode = "plot"  # Options:
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
        "user_number": 300,
        "time": 100000,
        "grid" : True,
        "random_start": True,
        "epr_fraction": 0.3,      # fraction of 'time' run with pure EPR before CF (1.0 = pure EPR; 0 < x < 1 enables CF phase)
        "top_k": 10,                # number of similar neighbors used in CF phase
    }

    ##TOPK STILL NOT IN THE SAVING/PLOT PROCEDURE!!!!!1

    grid_params = {
        "cell_lenght": 101, #must be odd to have a central cell
        "cell_number": 101
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
        "time": 1000000,
        "grid" : True
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
        "time": 1000000,
        "user_number": 500,       #For correct displaying
        "grid": False,
        "random_start": True,      # must match the simulate_params value used
        "epr_fraction": 1,      
    }

    plot_list = [
        #"mobility",
        #"rg_distribution", #checked but still double plot
        #"freq_distribution", 
        #"two_radius", 
        #"s_over_time", 
        #"rg_over_time",
        #"aggregate_grid",   
        #"batch_grid",  ##Problems in saving due to max lenght titole 260 characters
        "batch_stats"     
    ]

    show = False #if True only shows the plots, if False only saves them to disk

    start_time = time.time()
    logger.info(f"Running mode: {mode}")

    # ==========================================================
    # MODE 1: SIMULATE only
    # ==========================================================
    if mode == "simulate":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        simulation.save_raw_data()

    # ==========================================================
    # MODE 2: ANALYSE only
    # ==========================================================
    elif mode == "analyse":

        simulation = model.simulation.Simulation()
        simulation.run_analysis(analyse_params, grid_params=grid_params)

    # ==========================================================
    # MODE 3: SIMULATE + ANALYSE (no double disk load)
    # ==========================================================
    elif mode == "simulate_and_analyse":

        simulation = model.simulation.Simulation()
        simulation.run(simulate_params, grid_params=grid_params)
        raw_run_folder = simulation.save_raw_data()
        simulation.run_analysis(simulate_params, raw_run_folder=raw_run_folder, grid_params=grid_params)

    # ==========================================================
    # MODE 4: PLOT only
    # ==========================================================
    elif mode == "plot":

        simulation = model.simulation.Simulation()
        simulation.load_analysis_data(plot_params, grid_params=grid_params, plot_list=plot_list)
        simulation.plot(plot_list, show=show)

    else:
        raise ValueError(
            "Invalid mode. Use: 'simulate', 'analyse', 'simulate_and_analyse', or 'plot'"
        )

    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()