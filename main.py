import logging

logger = logging.getLogger(__name__)
from model.utils import spatial_power_law_sampling

def main():
    spatial_power_law_sampling()

if __name__ =='__main__':

    #bringing level attention for logger to Info
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    main()
