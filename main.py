import logging
import model.simulation
import model.utils
import time
from pprint import pprint

logger = logging.getLogger(__name__)

#bringing level attention for logger to Info
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

def main():

    # ==========================================================
    # 1) Choose mode
    # ==========================================================

    mode = "simulate"     # options:
                                   # "simulate"
                                   # "load"

    # Parameters for simulation
    simulation_params = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.5,
        "b_param": 0.7,
        "user_number": 1,
        "time": 1000000
    }

    # Filters for loading saved datasets

    #Ignoring a filter --> None
    filters = {
        "ro": 0.1,
        "gamma": 0.2,
        "a_param": 0.5,
        "b_param": 0.7,
        "time": 1000000
    }

    plot_list = [
         #"mobility",
         #"rg_distribution",
         #"freq_distribution",
         #"two_radius"
         #"s_over_time"
         "rg_ over_time"
    ]

    start_time = time.time()
    logger.info(f"Running mode: {mode}")

    # ==========================================================
    # MODE 1: SIMULATE 
    # ==========================================================
    if mode == "simulate":

        simulation = model.simulation.Simulation(
            **simulation_params,
            plots_to_generate=plot_list,
            sim_params=simulation_params
        )
        simulation.run()
        simulation.save_users_parquet()


    # ==========================================================
    # MODE 2: LOAD
    # ==========================================================
    elif mode == "load":

        simulation = model.simulation.Simulation(
            **simulation_params,
            plots_to_generate=plot_list,
            sim_params=simulation_params,
            filters = filters
        )

        simulation.load_data()
        simulation.plot()


    else:
        raise ValueError(
            "Invalid mode. Use: 'simulate', 'load'"
        )

    # ==========================================================
    # END
    # ==========================================================
    elapsed = time.time() - start_time
    print(f"\nElapsed time: {elapsed:.4f} seconds\n")


if __name__ == "__main__":
    main()