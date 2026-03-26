import numpy as np
import pandas as pd
import model.utils 
import logging
from pprint import pprint
import model.grid

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
        Travel distance for exploration steps. None if returning or CF move.
    angle : float or None
        Travel angle for exploration steps. None if returning or CF move.
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
    make_decision_cf(neighbors)
        CF variant of make_decision. Uses EPR probability to choose E/R, then
        delegates to cf_exploring / cf_returning, falling back to EPR if needed.
    exploring()
        Executes an exploration step, selecting a new unvisited venue and updating state.
    returning()
        Executes a return step, selecting a previously visited venue based on visit frequency.
    cf_exploring(neighbors)
        CF exploration: picks a venue visited by a similar user but not yet by this user.
        Tries neighbors in weighted-random order; returns False if all are exhausted.
    cf_returning(neighbors)
        CF return: picks a venue visited by both a similar user and this user.
        Returns False (triggering EPR fallback) if no common venue exists.
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
    
    def __init__(self, uid, a_param, b_param, simulation, grid=False, initial_position=None):
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
        grid : bool or Grid-like, optional
            Whether to attach a Grid to this user (default False).
        initial_position : tuple of float, optional
            (x, y) Cartesian starting coordinates.  When None (default) the
            origin (0, 0) is used.  When provided the user's first venue and
            the grid cell counter are initialised at this position instead.

        Notes
        -----
        This method initializes movement- and history-related attributes,
        sets up internal data structures for venue indexing, and records the
        first exploration event in both histories.
        """
 

        self.uid = uid 
        self.a_param = a_param
        self.b_param = b_param

        # Resolve starting position: use provided value or default to origin
        _start = initial_position if initial_position is not None else (0, 0)

        self.has_grid = (grid is not False)
        if self.has_grid:
            self.grid = model.grid.Grid(
                cell_lenght=simulation.grid_cell_lenght,
                cell_number=simulation.grid_cell_number,
                initial_position=_start if initial_position is not None else None
            )

        _first_venue_name = self._position_to_cell_name(_start) if self.has_grid else "Venue_1"

        self.user_history = [["State(E/R)", "Departure","Departure_x_coord", "Departure_y_coord", "Destination","Destination_x_coord", "Destination_y_coord", "Distance", "Angle", "Delta_t", "Clock"]] 
        self.venue_history = [["Venue_name", "Number of visits", "Venue_x_coord", "Venue_y_coord"], [_first_venue_name, 1, _start[0], _start[1]]]
        self.num_distinct_venues = 1 ##lo sostituirò con num di elementi colonna history?
        self.simulation = simulation 
        self.state = None
 
        self.previous_position = None
        self.position = _start
        self.next_position = None

        self.departure = _first_venue_name
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



        self._venue_index = {_first_venue_name: 1}
        self._row_index = None

        self.first_exploration()


    def _position_to_cell_name(self, position):
        return self.grid.position_to_cell_name(position)
    

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
        if self.has_grid:
            self.grid.update_cell_value_given_position(self.position)

    def make_decision_cf(self, neighbors):
        """
        CF variant of make_decision. Uses the same EPR probability to decide
        between exploration and return, then delegates to the CF movement methods.
        Falls back to standard EPR movement when no CF-compatible venues are found.

        Parameters
        ----------
        neighbors : list of (User, float)
            Pre-computed list of (neighbor_user, similarity_weight) pairs,
            sorted by descending similarity, produced by Simulation._build_cf_data().

        Notes
        -----
        - E/R decision uses the same P_new as EPR.
        - CF exploration: venue from similar user not yet visited by this user.
          Fallback to EPR exploration if all neighbors are exhausted.
        - CF return: venue visited by both a selected similar user and this user.
          Fallback to EPR return if no common venue exists with the drawn neighbor.
        - delta_r and angle are set to None for CF moves (no spatial sampling).
        - delta_t is always sampled from the temporal power-law, same as EPR.
        """
        P_new = self.simulation.ro * self.num_distinct_venues ** (-self.simulation.gamma)

        if np.random.random_sample(size=None) >= P_new:
            if not self.cf_returning(neighbors):
                self.returning()
                logger.info(f"User {self.uid}: CF return fallback to EPR.")
        else:
            if not self.cf_exploring(neighbors):
                self.exploring()
                logger.warning(f"User {self.uid}: CF exploration fallback to EPR.")

        self.delta_t = model.utils.temporal_power_law_sampling(par=self.b_param).item()
        self.previous_position = self.position
        self.position = self.next_position
        self.departure = self.user_history[-1][self.user_history[0].index("Destination")]
        self.clock = self.clock + self.delta_t
        self.update_histories()
        if self.has_grid:
            self.grid.update_cell_value_given_position(self.position)

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
            self.destination = self._position_to_cell_name(self.next_position) if self.has_grid else f"Venue_{self.num_distinct_venues}"
            self._venue_index[self.destination] = len(self.venue_history) 

    def returning(self):
            """
            Executes a return step. Selects a previously visited venue using
            frequency-weighted probability, sets the user state to returning,
            assigns the next position, and updates the destination label.
            """
            self.state = "R"
            self.next_position = self.get_return_position()
            self.destination = self.next_venue_return
            self.delta_r = None
            self.angle = None    

    def cf_exploring(self, neighbors):
        """
        CF exploration step. Searches for a venue visited by a similar user that
        this user has not yet visited, selected proportionally to the similar user's
        visit frequency for that venue.

        Neighbors are tried in a single weighted-random permutation (without
        replacement), so each neighbor is attempted at most once per call.
        Returns False when all neighbors have been tried without finding a valid
        venue, triggering an EPR exploration fallback in make_decision_cf().

        Parameters
        ----------
        neighbors : list of (User, float)
            (neighbor_user, similarity_weight) pairs.

        Returns
        -------
        bool
            True if a CF venue was found and state was updated; False otherwise.

        Notes
        -----
        On success sets: state="E", num_distinct_venues, next_position,
        destination, _venue_index entry, delta_r=None, angle=None.
        """
        if not neighbors:
            return False

        my_venues = set(self._venue_index.keys())

        neighbor_users = [u for u, _ in neighbors]
        weights = np.array([w for _, w in neighbors], dtype=float)
        if weights.sum() == 0:
            logger.warning(f"User {self.uid}: CF exploration skipped due to zero similarity weights.")
            return False
        probs = weights / weights.sum()

        # Weighted random permutation: try each neighbor at most once
        order = np.random.choice(len(neighbor_users), size=len(neighbor_users),
                                 replace=False, p=probs)

        for idx in order:
            neighbor = neighbor_users[idx]
            # Venues visited by this neighbor but not yet by current user
            candidate_rows = [row for row in neighbor.venue_history[1:]
                              if row[0] not in my_venues]
            if not candidate_rows:
                continue

            visit_counts = np.array([row[1] for row in candidate_rows], dtype=float)
            p = visit_counts / visit_counts.sum()
            chosen = candidate_rows[np.random.choice(len(candidate_rows), p=p)]

            self.state = "E"
            self.num_distinct_venues += 1
            self.next_position = (chosen[2], chosen[3])
            self.destination = chosen[0]
            self._venue_index[self.destination] = len(self.venue_history)#?
            self.delta_r = None
            self.angle = None
            return True

        return False

    def cf_returning(self, neighbors):
        """
        CF return step. Selects a venue visited by both a randomly drawn similar
        user and this user (excluding the current position), chosen proportionally
        to this user's own visit frequency for those venues.

        Only one neighbor is sampled. If no common venue exists with that neighbor,
        returns False immediately, triggering an EPR return fallback in
        make_decision_cf().

        Parameters
        ----------
        neighbors : list of (User, float)
            (neighbor_user, similarity_weight) pairs.

        Returns
        -------
        bool
            True if a CF venue was found and state was updated; False otherwise.

        Notes
        -----
        On success sets: state="R", next_position, next_venue_return,
        _row_index, delta_r=None, angle=None.
        """
        if not neighbors:
            logger.info(f"User {self.uid}: CF return skipped due to no neighbors.")
            return False

        neighbor_users = [u for u, _ in neighbors]
        weights = np.array([w for _, w in neighbors], dtype=float)
        if weights.sum() == 0:
            logger.info(f"User {self.uid}: CF return skipped due to zero similarity weights.")
            return False
        probs = weights / weights.sum()

        idx = np.random.choice(len(neighbor_users), p=probs)
        neighbor = neighbor_users[idx]

        neighbor_venue_names = {row[0] for row in neighbor.venue_history[1:]}

        # Venues visited by both this user and the chosen neighbor, excluding current position
        common_rows = [row for row in self.venue_history[1:]
                       if row[0] in neighbor_venue_names
                       and (row[2], row[3]) != self.position]

        if not common_rows:
            return False

        visit_counts = np.array([row[1] for row in common_rows], dtype=float)
        p = visit_counts / visit_counts.sum()
        chosen = common_rows[np.random.choice(len(common_rows), p=p)]

        self.state = "R"
        self.next_position = (chosen[2], chosen[3])
        self.next_venue_return = chosen[0]
        self._row_index = self._venue_index[chosen[0]]
        self.delta_r = None
        self.angle = None
        return True

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
        self.next_venue_return = venue_ids[choice]

        col_x_index = self.venue_history[0].index("Venue_x_coord")
        col_y_index = self.venue_history[0].index("Venue_y_coord")

        self._row_index = self._venue_index[self.next_venue_return]

        return self.venue_history[self._row_index][col_x_index], self.venue_history[self._row_index][col_y_index]

    
    def get_exploration_venue(self):
        """
        Generates a new exploration destination using spatial power-law sampling.

        If a grid is active (has_grid=True), samples candidate positions until one
        falls in an unvisited cell, then marks that cell as visited.
        If no grid is active, a single sample is drawn unconditionally.

        Returns
        -------
        tuple of float
            (x, y) Cartesian coordinates of the chosen exploration destination.
        """
        if self.has_grid:
            saturation = 0
            while True:
                self.delta_r = model.utils.spatial_power_law_sampling(par=self.a_param).item()
                self.angle = np.random.uniform(0, 2 * np.pi)
                next_position = model.utils.polar_to_cartesian(self.delta_r, self.angle, position=self.position)
                cell_value = self.grid.get_cell_value_given_position(next_position)
                if cell_value is not False and cell_value == 0:
                    break
                saturation += 1
                if saturation > 100000:
                    raise RuntimeError(f"User {self.uid}: Unable to find unvisited cell after 100000 attempts. Check grid parameters or user density.")
        else:
            self.delta_r = model.utils.spatial_power_law_sampling(par=self.a_param).item()
            self.angle = np.random.uniform(0, 2 * np.pi)
            next_position = model.utils.polar_to_cartesian(self.delta_r, self.angle, position=self.position)
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
        if self.has_grid:
            self.grid.update_cell_value_given_position(self.position)

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