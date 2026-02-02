import numpy as np
import model.user
import model.utils
import pandas as pd
import os
from datetime import datetime
from matplotlib import pyplot as plt
from pprint import pprint

import logging
logger = logging.getLogger(__name__)

class Simulation:
    
    def __init__(self, ro, gamma, a_param, b_param, user_number, time, plots_to_generate=None, sim_params=None, filters = None):
                 #ro_filter, gamma_filter, a_param_filter, b_param_filter, time_filter):
        self.ro = ro
        self.gamma = gamma
        self.a_param = a_param
        self.b_param = b_param
        self.user_number = user_number
        self.time = time
        self.plots_to_generate = plots_to_generate or []
        self.sim_params = sim_params or {}
        self.user = None
        self.rg = None
        self.users_list = []
        self.user_history_list = []
        self.rg_list = []
        self.two_rg_list = []
        self.filters = filters

        self.last_user_history = None
        self.last_venue_history = None
        self.last_stats = None
        self.last_x_cm = None
        self.last_y_cm = None

    def run(self):

        for i in range(self.user_number):
            self.user = model.user.User(uid = f"User_{i}", a_param=self.a_param, b_param=self.b_param, simulation = self)
            while self.user.clock < self.time:
                self.user.make_decision()
            self.user.calculate_statistics()

            self.users_list.append(self.user)
            self.rg_list.append(self.user.r_gyr)
            self.two_rg_list.append(self.user.r_gyr_two)

    def load_data(self):
        logger.info("Loading saved simulation data...")
        loaded_users = model.utils.load_simulations_parquet(filters=self.filters)
        #print((loaded_users))

        for entry in loaded_users:
            # entry is the tuple (df1, df2, stats)
            self.last_user_history = entry[0]
            self.last_venue_history = entry[1]
            self.last_stats = entry[2]

            if self.last_stats is not None:
                rg_value = self.last_stats.loc[self.last_stats['metric'] == "radius of gyration", 'value'].iloc[0]
                rg_two_value = self.last_stats.loc[self.last_stats['metric'] == "radius of gyration two", 'value'].iloc[0]
                self.last_x_cm = self.last_stats.loc[self.last_stats['metric'] == "x_center of mass", 'value'].iloc[0]
                self.last_y_cm = self.last_stats.loc[self.last_stats['metric'] == "y_center of mass", 'value'].iloc[0]
                self.rg_list.append(rg_value) 
                self.two_rg_list.append(rg_two_value)

            self.user_history_list.append(entry[0])



    def save_users_parquet(self, base_folder=None):
        """
        Save user history, venue history and stats to Parquet files.

        Folder structure:
        base_folder/
            ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                {unique_run_id}/
                    user_0_history.parquet
                    user_0_venue.parquet
                    ...
        """

        logger.info("Saving simulation results...")

    # Default folder: sata/ at project root
        if base_folder is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_folder = os.path.join(project_root, "data")
        os.makedirs(base_folder, exist_ok=True)

        # Folder name based on simulation parameters (excluding user_number)
        folder_name = f"ro_{self.ro}_gamma_{self.gamma}_a_{self.a_param}_b_{self.b_param}_time_{self.time}"
        folder_path = os.path.join(base_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Unique subfolder for this execution to prevent overwriting
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_folder_path = os.path.join(folder_path, run_id)
        os.makedirs(run_folder_path, exist_ok=True)

        for i, user in enumerate(self.users_list):
            # Save user_history
            user_df = pd.DataFrame(user.user_history[1:], columns=user.user_history[0])
            user_df.to_parquet(os.path.join(run_folder_path, f"user_{i}_history.parquet"), index=False)

            # Save venue_history
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])
            venue_df.to_parquet(os.path.join(run_folder_path, f"user_{i}_venue.parquet"), index=False)

            # Save stats if they exist
            if hasattr(user, "stats") and user.stats is not None:
                stats_df = pd.DataFrame(user.stats[1:], columns=user.stats[0])  # skip header
                stats_df.to_parquet(os.path.join(run_folder_path, f"user_{i}_stats.parquet"), index=False)

        logger.info("Simulation results saved...")

    def plot(self):

        if "mobility" in self.plots_to_generate:
            model.utils.plot_mobility_visualization(
            self.last_user_history,
            self.last_venue_history,
            self.last_stats,
            rg=True,
            title=model.utils.build_plot_title("User Mobility", self.sim_params)
            )

        if "rg_distribution" in self.plots_to_generate:
            model.utils.plot_rg_distribution(
            self.rg_list,
            #num_bins=20,
            title=model.utils.build_plot_title("RG Distribution", self.sim_params)
            )

        if "freq_distribution" in self.plots_to_generate:
            model.utils.plot_frequency_distribution(
            self.last_venue_history,
            self.gamma,
            title=model.utils.build_plot_title("Frequency Distribution", self.sim_params)
            )


        if "two_radius" in self.plots_to_generate:
            model.utils.plot_two_radius_vs_radius(
            self.rg_list,
            self.two_rg_list,
            title=model.utils.build_plot_title("Two-RG vs RG", self.sim_params)
            )

        if "s_over_time" in self.plots_to_generate:
            model.utils.plot_S_over_time(
            self.user_history_list,
            bins_per_decade=10,
            title=model.utils.build_plot_title("S over time", self.sim_params),
            k = 100
            )

        if "rg_ over_time" in self.plots_to_generate:
            model.utils.plot_r_gyr_over_time(
                self.user_history_list,
                title=None,
                bins_per_decade=100,
                k=100)
        
            
 
        




        


        