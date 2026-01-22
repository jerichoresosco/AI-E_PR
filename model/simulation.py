import numpy as np
from model.user import User
from model.utils import compute_exploration_points, plot_mobility_visualization, trajectory_from_user, radius_of_gyration
from pprint import pprint
import pandas as pd
from matplotlib import pyplot as plt

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
            self.user.convert_histories_to_df()
            self.users_list.append(self.user)
            trajectory = trajectory_from_user(self.user)
            pprint(trajectory)
            x, y, sizes, x_traj, y_traj = compute_exploration_points(self.user)
            plot_mobility_visualization(x,y,sizes,x_traj, y_traj, rg = True)
            _, rg = radius_of_gyration(x_traj, y_traj)
            pprint(f"Radius of gyration: {rg}")
            
            pprint(self.user.user_history)
 
        




        


        