import numpy as np
import pandas as pd
import model.utils 
import logging
from pprint import pprint

logger = logging.getLogger(__name__)

class User:
    """
    Represents a user within the mobility simulation. The User class encodes the user's
    movement logic, state transitions, and the storage of mobility-related information.

    Attributes
    ----------
    uid : int
        Unique identifier of the user.
    a_param : float
        Spatial exploration power-law parameter.
    b_param : float
        Temporal power-law parameter.
    user_history : list
        Sequential log of user states and movements, including state (E/R), origin,
        destination, coordinates, distance, angle, waiting time, and clock time.
    venue_history : list
        List of visited venues, storing venue name, visit count, and coordinates.
    simulation : Simulation
        Reference to the parent simulation instance providing parameters.
    num_distinct_venues : int
        Number of distinct venues the user has explored.
    state : str
        Current user state: "E" (exploring) or "R" (returning).
    previous_position : tuple or None
        Coordinates of the user's previous position.
    position : tuple
        Coordinates of the user's current position.
    next_position : tuple or None
        Coordinates of the user's next selected position.
    departure : str
        Name of the departure venue for the current movement.
    destination : str or None
        Name of the destination venue for the current movement.
    previous_destination : str or None
        Name of the previous destination venue.
    next_venue_return : int or None
        Index of the venue selected for a return movement.
    delta_r : float or None
        Travel distance for exploration steps. None if returning.
    angle : float or None
        Travel angle for exploration steps. None if returning.
    delta_t : float or None
        Waiting time at the current venue.
    clock : float
        Accumulated simulation time for the user.
    r_cm : np.ndarray or None
        Center-of-mass coordinates of the user trajectory.
    r_cm_two : np.ndarray or None
        Center-of-mass coordinates of the top-2 most visited venues.
    r_gyr : float or None
        Radius of gyration of the full trajectory.
    r_gyr_two : float or None
        Radius of gyration computed using only the top-2 most visited venues.
    stats : list
        Tabular list of computed trajectory statistics.
    _venue_index : dict
        Mapping from venue names to their row index inside venue_history.
    _row_index : int or None
        Cached row index for updating venue visit counts.

    Methods
    -------
    __init__(uid, a_param, b_param, simulation)
        Initializes the user and performs the first exploration movement.
    make_decision()
        Determines whether the user explores or returns, updates movement,
        waiting time, clock, and histories.
    exploring()
        Executes an exploration step, selecting a new unvisited venue and updating state.
    returning()
        Executes a return step, selecting a previously visited venue based on visit frequency.
    get_return_position()
        Selects a return venue using a frequency-weighted probability distribution and
        returns its coordinates.
    get_exploration_venue()
        Generates a new exploration destination using spatial power-law sampling.
    first_exploration()
        Performs the mandatory initial exploration movement and initializes histories.
    update_histories()
        Updates user_history and venue_history according to the current movement and state.
    calculate_statistics()
        Computes trajectory descriptors such as center of mass and radius of gyration,
        storing results in the stats table.
    """
    
    def __init__(self, uid, a_param, b_param, simulation):
        """
        Initializes a new User instance, sets default attributes, internal indices,
        initial position, and immediately performs the first mandatory exploration step.

        Parameters
        ----------
        uid : int
            Unique identifier for the user.
        a_param : float
            Spatial exploration power-law parameter.
        b_param : float
            Temporal power-law parameter.
        simulation : Simulation
            Reference to the parent simulation object providing model parameters.

        Notes
        -----
        This method initializes movement- and history-related attributes,
        sets up internal data structures for venue indexing, and records the
        first exploration event in both histories.
        """
 

        self.uid = uid 
        self.a_param = a_param
        self.b_param = b_param


        self.user_history = [["State(E/R)", "Departure","Departure_x_coord", "Departure_y_coord", "Destination","Destination_x_coord", "Destination_y_coord", "Distance", "Angle", "Delta_t", "Clock"]] 
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
        Determines whether the user will explore a new venue or return to a previously
        visited one based on the exploration probability. Updates the movement state,
        computes waiting time, refreshes the simulation clock, updates positions,
        and records the movement in the histories.

        Notes
        -----
        Exploration probability is computed as:
            P_new = ro * (num_distinct_venues)^(-gamma)
        provided by the simulation model.

        The method:
        - chooses exploration or return
        - samples waiting time (delta_t)
        - updates departure, position, and clock
        - logs the transition using update_histories() 

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
            Executes an exploration step. Selects a new unvisited venue by sampling
            a spatial displacement and angle, increments the number of distinct venues,
            sets the next position, and updates the destination label.

            Notes
            -----
            The venue index is updated to map the new venue name to its row in the
            venue history. The actual exploration movement (delta_r, angle) is
            generated later inside get_exploration_venue().
            """
            self.state = "E"
            self.num_distinct_venues += 1
            self.next_position = self.get_exploration_venue()
            self.destination = f"Venue_{self.num_distinct_venues}"
            self._venue_index[f"Venue_{self.num_distinct_venues}"] = len(self.venue_history) 

    def returning(self):
            """
            Executes a return step. Selects a previously visited venue using
            frequency-weighted probability, sets the user state to returning,
            assigns the next position, and updates the destination label.
            """
            self.state = "R"
            self.next_position = self.get_return_position()
            self.destination = f"Venue_{self.next_venue_return}"
            self.delta_r = None
            self.angle = None    

    def get_return_position(self):
        """
        Selects a previously visited venue (excluding the current one) using a
        probability distribution proportional to the number of past visits.
        Returns the coordinates of the chosen venue.

        Returns
        -------
        tuple of float
            (x_coord, y_coord) of the selected return venue.

        Notes
        -----
        - Candidate venues are all visited venues except the current one.
        - Probabilities are computed as:
            p_i = visits_i / sum(visits)
        - The internal index _row_index is set to the selected venue row in venue_history.
        """
        

        candidates = [row for row in self.venue_history[1:] if (row[2], row[3]) != self.position]

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
        """"
        Computes trajectory-based metrics, including:
        - center of mass (full trajectory)
        - center of mass using top-2 most visited venues
        - radius of gyration (full)
        - radius of gyration using top-2 venues
        """

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
                
    