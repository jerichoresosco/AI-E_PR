import numpy as np
from model.user import User

import logging
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(self, ro, gamma, time):
        logger.info("Initializing the simulation...")
        self.ro = ro
        self.gamma = gamma
        self.time = time

    def run(self):
        user = User(uid = "Pippo", simulation = self)
        print(user.uid)
        while user.clock < self.time:
            user.make_decision()

        