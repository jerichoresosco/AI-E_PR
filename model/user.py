import numpy as np
import model.utils 
import logging
import model.grid
import model.utils_cf

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
    d_weights : np.ndarray or None
        Global, shared d-EPR attractiveness matrix (shape (cell_number,
        cell_number)), identical across every user in the simulation. When
        not None, exploration uses the gravity-model choice implemented in
        _gravity_exploration_venue() instead of spatial power-law sampling.
        None for "ce"/"ge" (classic/grid EPR).

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
    _gravity_exploration_venue()
        d-EPR variant of get_exploration_venue(): chooses the next venue among
        all unvisited cells using a gravity-model probability law instead of
        spatial power-law sampling. Used when d_weights is not None.
    first_exploration()
        Performs the mandatory initial exploration movement and initializes histories.
    update_histories()
        Updates user_history and venue_history according to the current movement and state.
    calculate_statistics()
        Computes trajectory descriptors such as center of mass and radius of gyration,
        storing results in the stats table.
    """
    
    def __init__(self, uid, a_param, b_param, simulation, grid=False, initial_position=None, d_weights=None):
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
        d_weights : np.ndarray or None, optional
            Global, shared d-EPR attractiveness matrix (shape (cell_number,
            cell_number)). When provided, exploration uses the d-EPR
            gravity-model choice instead of spatial power-law sampling
            (requires grid to be active). Default None (classic/grid EPR).

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

        # When the grid is active, snap the starting position to the centroid of
        # the corresponding cell so that all users share the same discrete venues
        # from the very first step.
        if grid is not False:
            _start = model.grid.Grid.snap_to_centroid_static(
                _start,
                simulation.grid_cell_lenght,
                simulation.grid_cell_number,
            )

        self.has_grid = (grid is not False)
        if self.has_grid:
            self.grid = model.grid.Grid(
                cell_lenght=simulation.grid_cell_lenght,
                cell_number=simulation.grid_cell_number,
                initial_position=_start if initial_position is not None else None
            )

        # d-EPR support: when a global weight matrix is supplied, exploration
        # switches from spatial power-law sampling to a gravity-model choice
        # among all unvisited cells (see get_exploration_venue() /
        # _gravity_exploration_venue()).
        self.d_weights = d_weights

        _first_venue_name = self._position_to_cell_name(_start) if self.has_grid else "Venue_1"

        self.user_history = [["State(E/R)", "Departure","Departure_x_coord", "Departure_y_coord", "Destination","Destination_x_coord", "Destination_y_coord", "Distance", "Angle", "Delta_t", "Clock"]] 
        self.venue_history = [["Venue_name", "Number of visits", "Venue_x_coord", "Venue_y_coord", "Total time"], [_first_venue_name, 1, _start[0], _start[1], 0.0]]
        self.num_distinct_venues = 1 ##lo sostituirò con num di elementi colonna history?
        self.simulation = simulation 
        self.state = None
 
        self.previous_position = None
        self.position = _start
        self.next_position = None

        self.departure = _first_venue_name
        self.destination = None

        self.next_venue_return = None
        self.delta_r = None
        self.angle = None
        self.delta_t = None
        self.clock = 0
        self.r_gyr = None

        self.stats = [
                    ["metric", "value"],
        ]



        self._venue_index = {_first_venue_name: 1}
        self._row_index = None

        self.first_exploration()


    def _position_to_cell_name(self, position):
        return self.grid.position_to_cell_name(position)

    @property
    def _spatial_bounds(self):
        """(min_val, max_val) for spatial (jump-length) power-law sampling.

        Read from the attached simulation/stub (defaults POWERLAW_DEFAULTS).
        """
        sim, d = self.simulation, model.utils.POWERLAW_DEFAULTS
        return (getattr(sim, "spatial_min", d["spatial_min"]),
                getattr(sim, "spatial_max", d["spatial_max"]))

    @property
    def _temporal_bounds(self):
        """(min_val, max_val) for temporal (waiting-time) power-law sampling."""
        sim, d = self.simulation, model.utils.POWERLAW_DEFAULTS
        return (getattr(sim, "temporal_min", d["temporal_min"]),
                getattr(sim, "temporal_max", d["temporal_max"]))
    

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

        _tmin, _tmax = self._temporal_bounds
        self.delta_t = model.utils.temporal_power_law_sampling(
            par=self.b_param, min_val=_tmin, max_val=_tmax).item()
        self.previous_position = self.position
        self.position = self.next_position
        self.departure = self.user_history[-1][self.user_history[0].index("Destination")]
        self.clock = self.clock + self.delta_t
        self.update_histories()
        if self.has_grid:
            self.grid.update_cell_value_given_position(self.position)

    def make_decision_cf(self, neighbors, cf_type="uk"):
        """
        CF variant of make_decision.

        Parameters
        ----------
        neighbors : cf_type-specific structure
            "uk" / "ik" / "mf" -> list of (venue_row, score)
        cf_type : str
            "uk" (User-KNN), "ik" (Item-KNN), "mf" (Matrix Factorisation), ...
            Default "uk".
        """
        model.utils_cf.make_decision_cf(self, neighbors, cf_type)
        

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
            if self.next_position is None:
                # Grid saturated: there is no unvisited cell left to explore.
                # Undo the exploration bookkeeping and fall back to a return —
                # the natural EPR behaviour when nothing new is reachable. A
                # saturated grid means many venues have been visited, so
                # returning() always has candidates.
                self.num_distinct_venues -= 1
                self.returning()
                return
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
            
            dx = self.next_position[0] - self.position[0]
            dy = self.next_position[1] - self.position[1]
            distance = np.sqrt(dx**2 + dy**2)

            self.delta_r = distance
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
        self.next_venue_return = venue_ids[choice]

        col_x_index = self.venue_history[0].index("Venue_x_coord")
        col_y_index = self.venue_history[0].index("Venue_y_coord")

        self._row_index = self._venue_index[self.next_venue_return]

        return self.venue_history[self._row_index][col_x_index], self.venue_history[self._row_index][col_y_index]

    
    def get_exploration_venue(self):
        """
        Generates a new exploration destination.

        If d_weights is set (d-EPR), delegates to _gravity_exploration_venue(),
        which picks the next venue among all unvisited cells using a
        gravity-model probability law instead of spatial power-law sampling.

        Otherwise, if a grid is active (has_grid=True), samples candidate
        positions using spatial power-law sampling until one falls in an
        unvisited cell. The actual destination is then snapped to the
        centroid of that cell, and delta_r and angle are recomputed accordingly so
        that they remain consistent with the snapped position. The angle is always
        returned in [0, 2*pi].
        If no grid is active, a single sample is drawn unconditionally.

        Returns
        -------
        tuple of float
            (x, y) Cartesian coordinates of the chosen exploration destination.
        """
        if self.d_weights is not None:
            next_position = self._gravity_exploration_venue()
        elif self.has_grid:
            # Saturation guard (O(1)): on the grid, num_distinct_venues counts
            # the distinct visited cells, so it equals the total cell count iff
            # every cell has been visited. When that happens there is nothing
            # new to explore -> signal the caller (returns None) instead of
            # spinning the sampling loop and raising.
            total_cells = self.grid.matrix.shape[0] * self.grid.matrix.shape[1]
            if self.num_distinct_venues >= total_cells:
                return None
            _smin, _smax = self._spatial_bounds
            saturation = 0
            while True:
                self.delta_r = model.utils.spatial_power_law_sampling(
                    par=self.a_param, min_val=_smin, max_val=_smax).item()
                self.angle = np.random.uniform(0, 2 * np.pi)
                next_position = model.utils.polar_to_cartesian(self.delta_r, self.angle, position=self.position)
                cell_value = self.grid.get_cell_value_given_position(next_position)
                if cell_value is not False and cell_value == 0:
                    # Snap to the centroid of the selected cell so that all users
                    # share the same discrete set of venues.
                    cell_x, cell_y = self.grid._position_to_cell_coords(next_position)
                    next_position = self.grid.cell_coords_to_centroid(cell_x, cell_y)
                    # Recompute delta_r and angle to be consistent with the centroid.
                    dx = next_position[0] - self.position[0]
                    dy = next_position[1] - self.position[1]
                    self.delta_r = np.sqrt(dx**2 + dy**2)
                    self.angle = np.arctan2(dy, dx) % (2 * np.pi)
                    break
                saturation += 1
                if saturation > 100000:
                    # Safety net: unvisited cells may still exist but the
                    # movement law can't reach them by sampling. Rather than
                    # raise, give up exploring this step and let the caller
                    # fall back to a return.
                    return None
        else:
            _smin, _smax = self._spatial_bounds
            self.delta_r = model.utils.spatial_power_law_sampling(
                par=self.a_param, min_val=_smin, max_val=_smax).item()
            self.angle = np.random.uniform(0, 2 * np.pi)
            next_position = model.utils.polar_to_cartesian(self.delta_r, self.angle, position=self.position)
        return next_position

    def _gravity_exploration_venue(self):
        """
        d-EPR exploration: gravity-model choice among every unvisited cell.

        Selects the next venue using a gravity-model probability law:

            p_ij = (1/Norm) * d_i * d_j / r_ij^2

        where i is the cell the user currently occupies, j ranges over
        *every* unvisited cell of the grid (exhaustive search, no radius
        cutoff), d_i / d_j are the global attractiveness weights shared by
        all users (self.d_weights), r_ij is the Euclidean distance between
        the centroids of i and j, and Norm makes the probabilities over all
        candidate j sum to 1.

        Because d_i is identical for every candidate j (the user only ever
        leaves from its current cell), it cancels out under normalisation;
        this implementation uses the algebraically-reduced form d_j / r_ij^2
        directly. This yields exactly the same probabilities as the full
        formula and avoids a spurious 0/0 when d_i happens to be zero.

        Returns
        -------
        tuple of float
            (x, y) Cartesian coordinates of the chosen venue, snapped to the
            centroid of the selected cell.

        Raises
        ------
        RuntimeError
            If every cell in the grid has already been visited.
        """
        matrix = self.grid.matrix
        unvisited = np.argwhere(matrix == 0)

        if unvisited.size == 0:
            # Grid saturated: no unvisited cell left. Signal the caller
            # (exploring() falls back to a return) instead of raising.
            return None

        cell_x = unvisited[:, 0]
        cell_y = unvisited[:, 1]
        cl = self.grid.cell_lenght
        sc = self.grid.spatial_center
        centroid_x = (cell_x + 0.5) * cl - sc
        centroid_y = (cell_y + 0.5) * cl - sc

        dx = centroid_x - self.position[0]
        dy = centroid_y - self.position[1]
        r_sq = dx ** 2 + dy ** 2

        d_j = self.d_weights[cell_x, cell_y]
        weights = d_j / r_sq
        total = weights.sum()

        if total <= 0:
            logger.warning(
                f"User {self.uid}: all candidate cells have zero d-EPR weight — "
                f"falling back to a uniform choice among unvisited cells."
            )
            probs = np.full(len(unvisited), 1.0 / len(unvisited))
        else:
            probs = weights / total

        choice_idx = np.random.choice(len(unvisited), p=probs)
        next_position = (float(centroid_x[choice_idx]), float(centroid_y[choice_idx]))

        sel_dx = next_position[0] - self.position[0]
        sel_dy = next_position[1] - self.position[1]
        self.delta_r = float(np.sqrt(sel_dx ** 2 + sel_dy ** 2))
        self.angle = float(np.arctan2(sel_dy, sel_dx) % (2 * np.pi))

        return next_position
    
    def first_exploration(self):
        """
        Initialize the first movement as an exploration, 
        because a person cannot return to the position they are leaving.

        """
        self.exploring()
        _tmin, _tmax = self._temporal_bounds
        self.delta_t = model.utils.temporal_power_law_sampling(
            par=self.b_param, min_val=_tmin, max_val=_tmax).item()
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
         
        # Update total time spent at the departure venue
        departure_idx = self._venue_index[self.departure]
        time_col = self.venue_history[0].index("Total time")
        self.venue_history[departure_idx][time_col] += self.delta_t

        if self.state == "E":
            self.venue_history.append([self.destination, 1, *self.next_position, 0.0])
            self.user_history.append([self.state, self.departure, *self.previous_position,
                                    self.destination, *self.position, self.delta_r, self.angle, self.delta_t, self.clock])
        
        else:
            self.venue_history[self._row_index][self.venue_history[0].index("Number of visits")] += 1
            self.user_history.append([self.state, self.departure, *self.previous_position,
                                    self.destination, *self.position, self.delta_r, self.angle, self.delta_t, self.clock])