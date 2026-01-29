import numpy as np
import pandas as pd
import model.utils 
import logging
from pprint import pprint

logger = logging.getLogger(__name__)

class User:
    """
    Represents a user in the simulation. This class contains the user-logic of movement and contains
    the methods which allows to store the mobility information.

    Attributes:
        uid (int): Unique identifier for the user.
        user_history (list): History of the user's state (E/R), starting venue, destination, distance, angle and time.
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
        r_cm (np.array): center of mass radius
        r_gyr (float): radius of gyration

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
    
    def __init__(self, uid, a_param, b_param, simulation):
        """
    Initializes a User instance with an ID and a reference to the simulation model.

    Args:
        uid (int): Unique identifier for the user.
        simulation (Simulation): The simulation model from which parameters are imported.
    """
 

        self.uid = uid 
        self.a_param = a_param
        self.b_param = b_param


        self.user_history = [["State(E/R)", "Departure","Departure_x_coord", "Departure_y_coord", "Destination","Destination_x_coord", "Destination_y_coord", "Distance", "Angle", "Time", "Clock"]] 
        self.venue_history = [["Venue_name", "Number of visits", "Venue_x_coord", "Venue_y_coord"], ["Venue_1", 1, 0, 0]]
        self.num_distinct_venues = 1 ##lo sostituirò con num di elementi colonna history?
        self.simulation = simulation 
        self.state = None
 
        self.previous_position = None
        self.position = (0,0)
        self.next_position = None

        self.departure = "Venue_1"
        self.destination = None
        self.previous_destination= None

        self.next_venue_return = None
        self.delta_r = None
        self.angle = None
        self.delta_t = None
        self.clock = 0
        self.r_cm = None
        self.r_gyr = None

        self.stats = [
                    ["metric", "value"],
        ]



        self._venue_index = {"Venue_1": 1}
        self._row_index = None

        self.first_exploration()


    def make_decision(self):
        """
        Determines whether the user will explore a new venue or return to a previously visited one
        and performs the updates 

        """
        P_new = self.simulation.ro * self.num_distinct_venues ** (-self.simulation.gamma) #Exploration probability

        if np.random.random_sample(size=None) >= P_new:
            self.returning()

        else:
            self.exploring()

        self.delta_t = model.utils.temporal_power_law_sampling(par=self.b_param).item()
        self.previous_position = self.position
        self.position = self.next_position
        self.departure = self.user_history[-1][self.user_history[0].index("Destination")]
        self.clock = self.clock + self.delta_t
        self.update_histories()

    def exploring(self):
            """
            Method for the exploration part

            """
            self.state = "E"
            self.num_distinct_venues += 1
            self.next_position = self.get_exploration_venue()
            self.destination = f"Venue_{self.num_distinct_venues}"
            self._venue_index[f"Venue_{self.num_distinct_venues}"] = len(self.venue_history) - 1

    def returning(self):
            """
            Method for the returning part

            """
            self.state = "R"
            self.next_position = self.get_return_position()
            self.destination = f"Venue_{self.next_venue_return}"
            self.delta_r = None
            self.angle = None    

    def get_return_position(self):
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

        col_x_index = self.venue_history[0].index("Venue_x_coord")
        col_y_index = self.venue_history[0].index("Venue_y_coord")

        self._row_index = self._venue_index[f"Venue_{self.next_venue_return}"]

        return self.venue_history[self._row_index][col_x_index], self.venue_history[self._row_index][col_y_index]

    
    def get_exploration_venue(self):
        """
        This method will contain the logic of exploration :)

        """
        self.delta_r = model.utils.spatial_power_law_sampling(par=self.a_param).item()
        self.angle = np.random.uniform(0, 2 * np.pi)
        next_position = model.utils.polar_to_cartesian(self.delta_r, self.angle)
        return next_position
    
    def first_exploration(self):
        """
        Initialize the first movement as an exploration, 
        because a person cannot return to the position they are leaving.

        """
        self.exploring()
        self.delta_t = model.utils.temporal_power_law_sampling(par=self.b_param).item()
        self.previous_position = self.position
        self.position = self.next_position
        self.clock += self.delta_t
        self.update_histories()

    def update_histories(self):
        """
        Update histories of venues and user depending on which state has been selected

        """
         
        if self.state == "E":
            self.venue_history.append([self.destination, 1, *self.next_position])
            self.user_history.append([self.state, self.departure, *self.previous_position,
                                    self.destination, *self.position, self.delta_r, self.angle, self.delta_t, self.clock])
        
        else:
            self.venue_history[self._row_index][self.venue_history[0].index("Number of visits")] += 1
            self.user_history.append([self.state, self.departure, *self.previous_position,
                                    self.destination, *self.position, self.delta_r, self.angle, self.delta_t, self.clock])
    
    def calculate_statistics(self):

        x_traj, y_traj = model.utils.get_trajectory_arrays(self, k_top=None)
        x_traj_two, y_traj_two = model.utils.get_trajectory_arrays(self, k_top=2)

        x_cm = np.mean(x_traj)
        y_cm = np.mean(y_traj)

        x_cm_two = np.mean(x_traj_two)
        y_cm_two = np.mean(y_traj_two)

        self.r_cm = np.array([x_cm, y_cm])
        self.r_cm_two = np.array([x_cm_two, y_cm_two])

        self.r_gyr = np.sqrt(np.mean((x_traj - x_cm)**2 + (y_traj - y_cm)**2))
        self.r_gyr_two = np.sqrt(np.mean((x_traj_two - x_cm_two)**2 + (y_traj_two - y_cm_two)**2))

        self.stats = [
                    ["metric", "value"],
                    ["x_center of mass", x_cm],
                    ["x_center of mass two", x_cm_two],
                    ["y_center of mass", y_cm],
                    ["y_center of mass two", y_cm_two],
                    ["radius of gyration", self.r_gyr],
                    ["radius of gyration two", self.r_gyr_two]
        ]
                
    