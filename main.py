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
    simulation = model.simulation.Simulation(ro = 0.1,
                                             gamma = +0.2,
                                             a_param = 0.55,
                                             b_param= 0.8,
                                             user_number=1,
                                             time=52*7*17*3600)
    simulation.run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Simulation terminated!")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    

if __name__ =='__main__':
    
    main()
