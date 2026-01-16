import logging
from model.simulation import Simulation

logger = logging.getLogger(__name__)

def main():
    simulation = Simulation(ro = 0.6, gamma = 0.21, time=5*17*3600)
    simulation.run()

if __name__ =='__main__':

    #bringing level attention for logger to Info
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    main()
