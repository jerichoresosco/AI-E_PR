import logging
from model.simulation import Simulation
import time

logger = logging.getLogger(__name__)

#bringing level attention for logger to Info
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

def main():
    start_time = time.time()
    simulation = Simulation(ro = 0.6, gamma = -0.21,user_number=100, time=5*17*3600)
    simulation.run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")


if __name__ =='__main__':
    
    main()
