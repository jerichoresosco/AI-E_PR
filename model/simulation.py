import numpy as np
from model.user import User
from pprint import pprint

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
        user.first_exploration()
        while user.clock < self.time:
            user.make_decision()
        pprint(user.user_history)
        pprint(user.venue_history)


        