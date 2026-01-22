import numpy as np
import pandas as pd
from model.utils import spatial_power_law_sampling, temporal_power_law_sampling
import logging
from pprint import pprint

logger = logging.getLogger(__name__)

class User:
    """
    Represents a user in the simulation. This class contains the user-logic of movement and contains
    the methods which allows to store the mobility information.

    Attributes:
        uid (int): Unique identifier for the user.
        user_history (list): History of the user's state, starting venue, destination, distance, angle and time.
        venue_history (list): History of venues visited and number of visits per venue.
        simulation (Simulation): Reference to the simulation model instance.
        num_distinct_venues (int): Number of distinct venues visited.
        state (str): Current state of the user ("E" for exploring, "R" for returning).
        next_venue (int): The next venue selected by the user.
        position (int): Current venue position.
        prevoius_postion (int): Previous position of user.
        delta_r (float or None): Distance of movement (None if returning).
        angle (float or None): Angle of movement when exploring (None if returning)
        delta_t (float or None): Waiting time, defined as the time a user spent at one location.
        clock (float): Simulation clock time for the user.

        Methods:
        __init__: Initialize attributes of the user-simulation.
        make_decision: Decide whether to explore or return and update state, position, and history.
        exploring: Perform an exploration move, updating state, venue count, and history.
        returning: Perform a return move, updating state and incrementing visit count.
        get_return_venue: Select a previously visited venue (excluding current) using frequency-based probability.
        get_exploration_venue: Return a not visited yet venue.
        first_exploration: Execute the first movement, which must be an exploration.
        update_histories: Update venue and user hisotries

    """
    
    def __init__(self, uid, simulation):
        """
    Initializes a User instance with an ID and a reference to the simulation model.

    Args:
        uid (int): Unique identifier for the user.
        simulation (Simulation): The simulation model from which parameters are imported.
    """
 

        self.uid = uid 


        self.user_history = [["State(E/R)", "Starting_Venue", "Destination", "Distance", "Angle", "Time"]] 
        self.venue_history = [["Venue", "Number of visits"], [1, 1]]
        self.num_distinct_venues = 1 ##lo sostituirò con num di elementi colonna history?
        self.position = 1 #Position is starting as 1 so "Venue" in self.venue_history is following the indentation:
                            # element 1: [1,z], element 2 : [2,x], ..., element k: [k,c]

        self.simulation = simulation 
        self.state = None
        self.next_venue = None
        self.previous_position = None
        self.delta_r = None
        self.angle = None
        self.delta_t = None
        self.clock = 0


    def make_decision(self):
        """
        Determines whether the user will explore a new venue or return to a previously visited one
        and performs the updates 

        """
        P_new = self.simulation.ro * self.num_distinct_venues ** self.simulation.gamma #Exploration probability

        if np.random.random_sample(size=None) >= P_new:
            self.returning()

        else:
            self.exploring()

        self.delta_t = temporal_power_law_sampling().item()
        self.previous_position = self.position
        self.position = self.next_venue
        self.clock = self.clock + self.delta_t
        self.update_histories()

    def exploring(self):
            """
            Method for the exploration part

            """
            self.state = "E"
            self.num_distinct_venues += 1
            self.next_venue = self.get_exploration_venue()
            self.delta_r = spatial_power_law_sampling().item()
            self.angle = np.random.uniform(0, 2 * np.pi)

    def returning(self):
            """
            Method for the returning part

            """
            self.state = "R"
            self.next_venue = self.get_return_venue()
            self.delta_r = None
            self.angle = None    

    def get_return_venue(self):
        """
        This method contains the return-logic to a venue, which as in the paper, is that the user
        prefers to return to a venue which has been visited frequently before. 
        Details of algorithm functioning below.


        Returns:
        --------
        the index of the selected previous position (type: int)
        """
        
        #We exclude the possibility to return to the current position
        candidates = [row for i, row in enumerate(self.venue_history[1:], start=1) if i != self.position]

        #Splitting candidates dataset in two parts
        num_visits = np.array([row[1] for row in candidates])
        venue_ids = [row[0] for row in candidates]
        
        #Calculating probability based on the frequency and making a choice
        prob = num_visits / num_visits.sum()
        choice = np.random.choice(len(venue_ids), p=prob)

        #Retriving the right number/position in the history matching with the choice in venue_ids
        self.next_venue_return = [i for i, row in enumerate(self.venue_history) if row[0] == venue_ids[choice]][0]
        return self.next_venue_return
    
    def get_exploration_venue(self):
        """
        This method will contain the logic of exploration :)

        """
        return self.num_distinct_venues
    
    def first_exploration(self):
        """
        Initialize the first movement as an exploration, 
        because a person cannot return to the position they are leaving.

        """
        self.exploring()
        self.delta_t = temporal_power_law_sampling().item()
        self.previous_position = self.position
        self.position = self.next_venue
        self.clock += self.delta_t
        self.update_histories()

    def update_histories(self):
        """
        Update histories of venues and user depending on which state has been selected

        """
         
        if self.state == "E":
            self.venue_history.append([self.num_distinct_venues, 1])
            self.user_history.append([self.state, self.previous_position, self.position, self.delta_r, self.angle, self.delta_t])
        
        else:
            self.venue_history[self.next_venue_return][1] += 1
            self.user_history.append([self.state, self.previous_position, self.position, self.delta_r,self.angle, self.delta_t])


    def convert_histories_to_df(self):
        """
        Convert the user history and venue history lists into Pandas DataFrames.        

        """
        #User_history
        columns_user = self.user_history[0]   
        data_user = self.user_history[1:]     
        self.user_history = pd.DataFrame(data_user, columns=columns_user)

        #venue_history
        columns_venue = self.venue_history[0]
        data_venue = self.venue_history[1:]    
        self.venue_history = pd.DataFrame(data_venue, columns=columns_venue)

         

    





