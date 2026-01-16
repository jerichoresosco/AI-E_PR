import numpy as np
import pandas as pd

from model.utils import spatial_power_law_sampling, temporal_power_law_sampling

class User:
    
    def __init__(self, uid, simulation):

        self.uid = uid #userID
        self.history = [] #history of user [[E, xy, r1, t1][E, yz, r2, t2]...[R, yk, rn, tn]] 
        self.simulation = simulation
        self.num_distinct_venues = 0
        self.state = None
        self.next_venue = 0
        self.position = 0
        self.delta_r = 0
        self.delta_t = 0
        self.clock = 0


    def make_decision(self):
        
        if np.random.random_sample(size=None) >= self.simulation.ro * self.num_distinct_venues ** self.simulation.gamma:
            self.state = "R"
            self.next_venue = 0 ###Da Cambiare!!!!!!!!

        else:
            self.state = "E"
            self.num_distinct_venues = self.num_distinct_venues + 1
            self.next_venue = self.num_distinct_venues
            self.delta_r = spatial_power_law_sampling().item()

        self.delta_t = temporal_power_law_sampling().item()
        self.history.append([self.state, self.position, self.next_venue, self.delta_r, self.delta_t])
        self.position = self.next_venue
        self.clock = self.clock + self.delta_t








