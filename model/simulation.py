import numpy as np
import model.user
import model.utils
import pandas as pd
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)

class Simulation:
    def __init__(self, ro, gamma, a_param, b_param, user_number, time):
        self.ro = ro
        self.gamma = gamma
        self.a_param = a_param
        self.b_param = b_param
        self.user_number = user_number
        self.time = time
        self.user = None
        self.rg = None
        self.users_list = []
        self.rg_list = []
        self.two_rg_list = []

    def run(self):

        for i in range(self.user_number):
            self.user = model.user.User(uid = f"User_{i}", a_param=self.a_param, b_param=self.b_param, simulation = self)
            while self.user.clock < self.time:
                self.user.make_decision()
            model.utils.convert_histories_to_df(self.user)
            self.users_list.append(self.user)
            self.rg_list.append(self.user.radius_of_gyration)
            self.two_rg_list.append(self.user.two_radius_of_gyration)

        model.utils.plot_mobility_visualization(self.user, rg = True)
        model.utils.plot_rg_distribution(self.rg_list, num_bins=20)
        model.utils.plot_frequency_distribution(self.user.venue_history, self.gamma)
        model.utils.plot_two_radius_vs_radius(self.rg_list, self.two_rg_list)


        
            
 
        




        


        