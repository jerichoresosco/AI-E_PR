import numpy as np
from model.user import User
from pprint import pprint

import logging
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(self, ro, gamma,user_number, time):
        logger.info("Initializing the simulation...")
        self.ro = ro
        self.gamma = gamma
        self.user_number = user_number
        self.time = time
        self.user = None
        self.users_list = []

    def run(self):

        for i in range(self.user_number):
            self.user = User(uid = f"User_{i}", simulation = self)
            self.user.first_exploration()
            while self.user.clock < self.time:
                self.user.make_decision()
            self.users_list.append(self.user)
        pprint(self.user.user_history)
        pprint(self.user.venue_history)

        print(type(self.user.user_history))
        print(type(self.user.venue_history))

        shape_user = (len(self.user.user_history), len(self.user.user_history[0]))
        shape_venue = (len(self.user.venue_history), len(self.user.venue_history[0]))

        print(shape_user)
        print(shape_venue)


        