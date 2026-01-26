import logging
import model.simulation
import time
from pprint import pprint

logger = logging.getLogger(__name__)

#bringing level attention for logger to Info
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

def main():
    logger.info("Initializing the simulation...")
    start_time = time.time()

    plot_list = [
        #"mobility",
        #"rg_distribution",
        #"freq_distribution",
        #"two_radius"
    ]

    params = {
        "ro": 0.6,
        "gamma": 0.21,
        "a_param": 0.55,
        "b_param": 0.8,
        "user_number": 700,
        "time": 52*7*17*3600
    }

    simulation = model.simulation.Simulation(
        **params,
        plots_to_generate=plot_list,
        sim_params=params
    )

    simulation.run()
    simulation.plot()

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    

if __name__ =='__main__':
    
    main()
