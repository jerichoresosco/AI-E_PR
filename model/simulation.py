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
        self.user = None

    def run(self):
        self.user = User(uid = "Pippo", simulation = self)
        self.user.first_exploration()
        while self.user.clock < self.time:
            self.user.make_decision()
        pprint(self.user.user_history)
        pprint(self.user.venue_history)

        print(type(self.user.user_history))
        print(type(self.user.venue_history))

        shape_user = (len(self.user.user_history), len(self.user.user_history[0]))
        shape_venue = (len(self.user.venue_history), len(self.user.venue_history[0]))

        print(shape_user)
        print(shape_venue)


        