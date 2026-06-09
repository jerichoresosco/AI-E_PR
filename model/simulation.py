import numpy as np
import model.user
import model.utils
import model.utils_plot
import model.utils_cf
import model.grid
import pandas as pd
import os
from datetime import datetime
from pprint import pprint

import logging
logger = logging.getLogger(__name__)


class _VenueSnapshot:
    """
    Lightweight proxy that exposes a frozen copy of a single user's
    venue_history to build_cf_data().

    build_cf_data() (and _build_uv_matrix inside it) only reads
    user.venue_history — it never touches user_history, grid, clock, or
    any other User attribute.  Passing a _VenueSnapshot instead of the
    live User object guarantees that every user in a CF interval is
    evaluated against the same temporal baseline: the state of
    venue_history at the *start* of that interval, before any user has
    been advanced.  This removes the ordering asymmetry that would
    otherwise arise from simulating users sequentially.

    Parameters
    ----------
    venue_history : list
        The live user.venue_history at the moment the snapshot is taken.
        A shallow copy of the outer list is stored; individual rows are
        not copied because build_cf_data only reads them.
    """
    __slots__ = ("venue_history",)

    def __init__(self, venue_history):
        # Shallow-copy the list so that rows appended during simulation
        # of this interval do not appear in the frozen view.
        self.venue_history = list(venue_history)


class Simulation:
    """
    three-phase mobility simulation pipeline:

    Phase 1 - SIMULATE : run the EPR model for N users, save raw trajectories to raw_data/
    Phase 2 - ANALYSIS : load raw trajectories, compute analysis quantities, save to analysis_data/
    Phase 3 - PLOT     : load analysis data and generate all requested plots

    Each phase is self-contained and receives only the parameters it needs:
        run(simulate_params, grid_params)   requires ro, gamma, a_param, b_param, user_number, time,
                                            algorithm ("ce"|"ge"|"uk"|"ik"|"mf")
                                            optional: top_k (default 10, used for uk/ik/mf)
                                            grid_params required when algorithm != "ce": cell_lenght, cell_number
        run_analysis(analyse_params, grid_params)
                                            requires ro, gamma, a_param, b_param, time, algorithm
                                            grid_params required when algorithm != "ce"
        load_analysis_data(plot_params)     requires ro, gamma, a_param, b_param, time, algorithm, [user_number]
        plot(plot_list)

    Additional entry points for chained simulations:
        resume(simulate_params, grid_params)
            Continue a simulation on users already loaded in memory (self.users_list must be
            populated either by run() or by _reconstruct_users_from_raw()).
            resume_params["time"] is interpreted as the *new* absolute clock target: users
            will run until their clock reaches this value.

        _reconstruct_users_from_raw(load_params, grid_params)
            Load a specific raw_data batch from disk and rebuild live User objects from
            the saved Parquet files, so that resume() can continue the simulation.
            load_params extends the standard parameter dict with:
                "batch_id" (str, optional) : timestamp folder name to load.
                    When absent, the most-recent batch is used automatically.
    """

    def __init__(self):
        # Populated during run() -- holds User objects
        self.users_list = []

        # Populated during load_analysis_data() -- used by plot()
        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []
        self.all_s_over_time = []
        self.mean_dist_list = []

        # Entropy lists -- populated during load_analysis_data()
        self.s_rand_list = []
        self.s_unc_list  = []
        self.s_real_list = []

        # Multi-series data for multipleplot ("mp") mode
        # Each element is a dict: {"label": str, "rg_list": [...], "two_rg_list": [...],
        #   "mean_dist_list": [...], "rg_distribution": [...], "all_rg_over_time": [...],
        #   "all_s_over_time": [...]}
        self._mp_series = []

        # Single-user references for mobility / freq_distribution plots
        self.last_user_history = None
        self.last_venue_history = None
        self.last_stats = None

        # Aggregate grids -- populated during load_analysis_data()
        self.aggregate_grid            = None
        self.aggregate_grid_user_count = 0
        self.batch_grids               = []

        # Batch-level stats -- populated during load_analysis_data()
        self.batch_stats_list = []

        # Set during run(), needed by User objects at decision time
        self.ro    = None
        self.gamma = None

        # Stored after run() for save_raw_data() folder naming
        self._simulate_params = None

        # Stored after load_analysis_data() for plot() titles/filters
        self._plot_params = None

        #Selecting saveplot or showplot
        self._show = None

    # ==========================================================================
    # PHASE 1: SIMULATE
    # ==========================================================================

    def run(self, simulate_params, grid_params=None):
        """
        Run the EPR mobility simulation from scratch and store results in self.users_list.

        Only EPR algorithms are accepted here. CF algorithms ("uk", "ik", "mf",
        "ba", "bv", "ra") require pre-existing trajectory data and must be launched
        via resume() after _reconstruct_users_from_raw().

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "user_number", "time".
            Must contain: "algorithm" -- one of:
              "ce"  : classic EPR (no grid)
              "ge"  : grid EPR
            CF algorithms ("uk", "ik", "mf", "ba", "bv", "ra") raise ValueError here.
        grid_params : dict, optional
            Required when algorithm is "ge". Must contain "cell_lenght" and
            "cell_number".
        """
        ro           = simulate_params["ro"]
        gamma        = simulate_params["gamma"]
        a_param      = simulate_params["a_param"]
        b_param      = simulate_params["b_param"]
        user_number  = simulate_params["user_number"]
        time         = simulate_params["time"]
        algorithm         = simulate_params["algorithm"]
        starting_position = simulate_params["starting_position"]
        top_k        = simulate_params.get("top_k", 10)

        _VALID_ALGORITHMS = ("ce", "ge", "uk", "ik", "mf", "ba", "bv", "ra")
        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"run(): unknown algorithm '{algorithm}'. "
                f"Must be one of: {_VALID_ALGORITHMS}"
            )

        if algorithm not in ("ce", "ge"):
            raise ValueError(
                f"run(): algorithm='{algorithm}' is a CF algorithm and cannot be used "
                f"to start a simulation from scratch. "
                f"Use 'ce' or 'ge' for run(), then load the saved batch and call "
                f"resume() with algorithm='{algorithm}'."
            )

        # Derive grid flag from algorithm
        grid_exists = algorithm != "ce"

        # When grid is disabled every user starts at (0,0) regardless of what
        # starting_position says.  Normalise to "centered" before storing params
        # so that the saved folder name reflects the actual simulation behaviour.
        if not grid_exists and starting_position != "centered":
            logger.warning(
                f"run(): algorithm='ce' (no grid) but starting_position='{starting_position}' — "
                f"all users will start at (0,0). Overriding to 'centered' for folder naming."
            )
            starting_position = "centered"
            simulate_params = {**simulate_params, "starting_position": "centered"}

        logger.info(f"Starting position: '{starting_position}'")

        # Store on self so User objects can access ro/gamma via self.simulation,
        # and save_raw_data() can build the correct folder name
        self.ro               = ro
        self.gamma            = gamma
        self.a_param          = a_param
        self.b_param          = b_param
        self._simulate_params = simulate_params
        self.grid_exists      = grid_exists
        self._starting_position = starting_position

        if grid_exists:
            self._grid_params = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]
            if starting_position in ("random", "normal"):
                self._spatial_center = (self.grid_cell_lenght * self.grid_cell_number) / 2
                #This definition is done following the grid logic, where the grid is defined from (0,0) to
                #(cell_lenght*cell_number, cell_lenght*cell_number), so the center is in the middle of the grid, not at (0,0)

        milestone = max(1, user_number // 10) #10% progress milestones for logging
        logger.info(
            f"Starting simulation for {user_number} users (algorithm='{algorithm}')..."
        )

        # ---------------------------------------------------------------
        # EPR loop: create users and run them to full time
        # ---------------------------------------------------------------
        for i in range(user_number):
            if self._starting_position == "random":
                sc = self._spatial_center
                init_pos = (
                    np.random.uniform(-sc, sc),
                    np.random.uniform(-sc, sc),
                )
            elif self._starting_position == "normal":
                sc = self._spatial_center
                sigma = sc / 3
                def _sample(sc=sc, sigma=sigma):
                    while True:
                        v = np.random.normal(0, sigma)
                        if -sc <= v <= sc:
                            return float(v)
                init_pos = (_sample(), _sample())
            else:
                init_pos = None

            user = model.user.User(
                uid=f"User_{i}",
                a_param=a_param,
                b_param=b_param,
                grid=grid_exists,
                simulation=self,
                initial_position=init_pos,
            )
            while user.clock < time:
                user.make_decision()

            self.users_list.append(user)

            if (i + 1) % milestone == 0 or (i + 1) == user_number:
                pct = ((i + 1) / user_number) * 100
                logger.info(f"EPR progress: {pct:.0f}% ({i+1}/{user_number} users)")

        # ---------------------------------------------------------------
        # Post-run: replace users with empty user_history
        # ---------------------------------------------------------------
        replacement_uid = user_number
        users_replaced = 0
        valid_users = []

        for user in self.users_list:
            if len(user.user_history) > 1:
                valid_users.append(user)
            else:
                logger.warning(
                    f"{user.uid}: user_history is empty — discarding and replacing."
                )
                while True:
                    if self._starting_position == "random":
                        sc = self._spatial_center
                        init_pos = (
                            np.random.uniform(-sc, sc),
                            np.random.uniform(-sc, sc),
                        )
                    elif self._starting_position == "normal":
                        sc = self._spatial_center
                        sigma = sc / 3
                        def _sample(sc=sc, sigma=sigma):
                            while True:
                                v = np.random.normal(0, sigma)
                                if -sc <= v <= sc:
                                    return float(v)
                        init_pos = (_sample(), _sample())
                    else:
                        init_pos = None

                    replacement = model.user.User(
                        uid=f"User_{replacement_uid}",
                        a_param=a_param,
                        b_param=b_param,
                        grid=grid_exists,
                        simulation=self,
                        initial_position=init_pos,
                    )
                    while replacement.clock < time:
                        replacement.make_decision()

                    replacement_uid += 1

                    if len(replacement.user_history) > 1:
                        valid_users.append(replacement)
                        users_replaced += 1
                        break
                    else:
                        logger.warning(
                            f"User_{replacement_uid - 1}: replacement also empty, retrying..."
                        )

        self.users_list = valid_users

        if users_replaced > 0:
            logger.info(
                f"Replaced {users_replaced} empty user(s) with valid replacements."
            )

        #TO BE SOLVED!!! Here we are trying to remove what is generated past the limit time, that is the last step for each user,
        #but this solution is not working properly, because what is updated in make_decision and make_decision_cf are user_history,
        #venue_history and grid.matrix, so we need to find a way to remove the last step from all of them. We are correctly removing user_history,
        #partially venue history and the grid is not changed at all.
        #for user in self.users_list:
            #header    = user.user_history[0]
            #clock_idx = header.index("Clock")
            #state_idx = header.index("State(E/R)")
            #if len(user.user_history) > 1 and user.user_history[-1][clock_idx] > time:
                #last_row = user.user_history.pop()
                #if last_row[state_idx] == "E":
                    #user.venue_history.pop()

    def resume(self, simulate_params, grid_params=None):
        """
        Continue the simulation on users already in self.users_list.

        Users are advanced until their clock reaches simulate_params["time"],
        which is interpreted as an *absolute* target clock, not an additional
        duration.  This means the value should be larger than the clock values
        the users already carry (i.e. the end-time of the previous run phase).

        The algorithm is taken from simulate_params["algorithm"] and must be
        one of: "ce", "ge", "uk", "ik", "mf", "ba", "bv", "ra".

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time", "algorithm".
            Optional: "top_k" (int, default 10) for CF algorithms.
            "starting_position" and "user_number" must NOT be included:
            they are inherited from the loaded simulation.
        grid_params : dict, optional
            When the simulation uses a grid, must contain "cell_lenght" and
            "cell_number".  When None (default) self._grid_params is reused
            if it exists.
        """
        if not self.users_list:
            raise RuntimeError(
                "resume() called but self.users_list is empty. "
                "Call run() or _reconstruct_users_from_raw() first."
            )

        ro           = simulate_params["ro"]
        gamma        = simulate_params["gamma"]
        a_param      = simulate_params["a_param"]   # kept for symmetry; not used in loop
        b_param      = simulate_params["b_param"]
        time_target  = simulate_params["time"]
        algorithm    = simulate_params["algorithm"]
        top_k        = simulate_params.get("top_k", 10)

        is_cf = algorithm not in ("ce", "ge")

        # grid and starting_position are inherited from the loaded simulation,
        # not taken from simulate_params.
        grid_exists       = self.grid_exists
        starting_position = self._starting_position

        # Inject into the params dict so that folder naming helpers (_make_run_folder,
        # _make_combined_run_folder) have access to them via self._simulate_params.
        simulate_params = {**simulate_params, "starting_position": starting_position}

        self.ro               = ro
        self.gamma            = gamma
        self._simulate_params = simulate_params

        # Resolve grid_params
        if grid_params is not None:
            self._grid_params     = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]
        elif getattr(self, "_grid_params", None) is not None:
            grid_params = self._grid_params  # reuse from previous run cause it must match the loaded state

        n_users   = len(self.users_list)
        milestone = max(1, n_users // 10)

        # Compute the effective start of this resume window from the users' clocks
        current_clocks = [u.clock for u in self.users_list]
        resume_start   = sum(current_clocks) / n_users   # approximate; each user may differ
        extra_time     = time_target - resume_start

        if extra_time <= 0:
            logger.warning(
                f"resume() called but time_target={time_target} is not larger than "
                f"the current mean user clock ({resume_start:.1f}). Nothing to do."
            )
            return

        logger.info(
            f"Resuming simulation: {n_users} users, "
            f"clock range [{min(current_clocks):.1f}, {max(current_clocks):.1f}] "
            f"→ target={time_target}, algorithm='{algorithm}'"
        )

        if not is_cf:
            # ---------------------------------------------------------------
            # Pure EPR resume
            # ---------------------------------------------------------------
            for i, user in enumerate(self.users_list):
                while user.clock < time_target: #for each user the clock is different and is the same as the last one in user_history.
                    user.make_decision()
                if (i + 1) % milestone == 0 or (i + 1) == n_users:
                    pct = ((i + 1) / n_users) * 100
                    logger.info(f"Resume EPR progress: {pct:.0f}% ({i+1}/{n_users} users)")

        else:
            # ---------------------------------------------------------------
            # CF resume: run collaborative filtering for the full resume window
            # ---------------------------------------------------------------
            cf_updates   = 4
            n_intervals  = cf_updates + 1
            cf_duration  = time_target - resume_start
            interval_len = cf_duration / n_intervals

            check_run_dir = None  # created on first interval, reused afterwards
            # Generate the batch timestamp once so that the check folder shares
            # the same ID as the raw/analysis batch saved for this run.
            batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._batch_ts = batch_ts   # store so save_raw_data / run_analysis can reuse it

            for interval in range(n_intervals):
                t_end = resume_start + (interval + 1) * interval_len
                logger.info(
                    f"Resume CF interval {interval + 1}/{n_intervals}: "
                    f"rebuilding similarities, running to {t_end:.1f}..."
                )

                # Freeze venue_history for all users at the start of this
                # interval so that build_cf_data sees a symmetric snapshot:
                # every user is represented by the same temporal baseline,
                # regardless of the order in which they will be simulated.
                frozen_users = [
                    _VenueSnapshot(user.venue_history)
                    for user in self.users_list
                ]

                cf_neighbors, check_run_dir = model.utils_cf.build_cf_data(
                    users_list     = frozen_users,
                    top_k          = top_k,
                    cf_type        = algorithm,
                    simulate_params= self._simulate_params,
                    grid_exists    = self.grid_exists,
                    grid_params    = getattr(self, "_grid_params", None),
                    check_run_dir  = check_run_dir,
                    interval_idx   = interval,
                    batch_ts       = batch_ts,
                )
                for i, user in enumerate(self.users_list):
                    neighbors = cf_neighbors[i]
                    while user.clock < t_end:
                        user.make_decision_cf(neighbors, algorithm)

        for user in self.users_list:
            header    = user.user_history[0]
            clock_idx = header.index("Clock")
            state_idx = header.index("State(E/R)")
            if len(user.user_history) > 1 and user.user_history[-1][clock_idx] > time_target:
                last_row = user.user_history.pop()
                if last_row[state_idx] == "E":
                    user.venue_history.pop()

        logger.info("Resume complete.")


    def _reconstruct_users_from_raw(self, load_params, grid_params=None):
        """
        Load a raw_data batch from disk and reconstruct live User objects so that
        resume() can continue the simulation from the saved state.

        The method locates the correct parameter folder in raw_data/ using
        load_params, then selects a specific batch sub-folder (by batch_id or
        automatically the most recent one), reads the Parquet files for every
        user, and rebuilds each User object in memory.

        Reconstructed state per user
        ----------------------------
        - user_history     : full list-of-lists, header row prepended
        - venue_history    : full list-of-lists, header row prepended
        - _venue_index     : rebuilt from venue_history
        - num_distinct_venues : len(venue_history) - 1
        - clock            : last Clock value in user_history
        - position         : (Destination_x_coord, Destination_y_coord) of last row
        - previous_position: (Departure_x_coord, Departure_y_coord) of last row
        - departure        : Departure field of last row
        - state            : State(E/R) field of last row
        - grid.matrix      : loaded from user_{i}_grid.parquet when grid is active
        - simulation       : reference to self (re-attached here)

        Parameters
        ----------
        load_params : dict
            Standard simulation parameter keys: "ro", "gamma", "a_param",
            "b_param", "time", "algorithm", "starting_position" (optional).
            Additional key:
                "batch_id" (str, optional)
                    Exact timestamp folder name to load (e.g. "20240601_120000_123456").
                    When absent, the most-recent batch folder is selected automatically.
        grid_params : dict, optional
            Required when load_params["algorithm"] is not "ce".
            Must contain "cell_lenght" and "cell_number".
            When None, self._grid_params is used if available.

        Returns
        -------
        str
            Path to the raw_data batch folder that was loaded, so it can be
            passed to run_analysis() or save_raw_data() if needed.

        Raises
        ------
        FileNotFoundError
            When the parameter folder or the requested batch folder does not exist.
        RuntimeError
            When no batch folders are found inside the parameter folder.
        """

        # Resolve grid_params
        if grid_params is not None:
            self._grid_params     = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]

        algorithm         = load_params["algorithm"]
        grid_exists       = algorithm != "ce"
        starting_position = load_params["starting_position"]
        batch_id          = load_params.get("batch_id", None)

        # When grid is disabled starting_position has no effect; normalise to
        # "centered" so the folder lookup matches what run() would have saved.
        if not grid_exists and starting_position != "centered":
            logger.warning(
                f"_reconstruct_users_from_raw(): algorithm='ce' (no grid) but "
                f"starting_position='{starting_position}' — overriding to 'centered' "
                f"for folder lookup."
            )
            starting_position = "centered"
            load_params = {**load_params, "starting_position": "centered"}

        # ------------------------------------------------------------------
        # Locate the parameter folder in raw_data/
        # ------------------------------------------------------------------
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, "raw_data")

        sp       = load_params["starting_position"]
        alg      = load_params["algorithm"]
        _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")

        if grid_exists and grid_params is not None:
            param_folder_name = (
                f"ro_{load_params['ro']}_g_{load_params['gamma']}"
                f"_a_{load_params['a_param']}_b_{load_params['b_param']}"
                f"_t_{load_params['time']}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_sp_{_sp_abbr}"
                f"_alg_{alg}"
            )
        else:
            param_folder_name = (
                f"ro_{load_params['ro']}_g_{load_params['gamma']}"
                f"_a_{load_params['a_param']}_b_{load_params['b_param']}"
                f"_t_{load_params['time']}"
                f"_sp_{_sp_abbr}"
                f"_alg_{alg}"
            )

        param_folder = os.path.join(base_folder, param_folder_name)
        if not os.path.exists(param_folder):
            raise FileNotFoundError(
                f"_reconstruct_users_from_raw: parameter folder not found:\n  {param_folder}"
            )

        # ------------------------------------------------------------------
        # Select the batch sub-folder
        # ------------------------------------------------------------------
        if batch_id is not None:
            run_folder = os.path.join(param_folder, batch_id)
            if not os.path.exists(run_folder):
                raise FileNotFoundError(
                    f"_reconstruct_users_from_raw: requested batch_id not found:\n  {run_folder}"
                )
        else:
            # Pick the most-recent batch
            candidates = sorted(
                d for d in os.listdir(param_folder)
                if os.path.isdir(os.path.join(param_folder, d))
            )
            if not candidates:
                raise RuntimeError(
                    f"_reconstruct_users_from_raw: no batch folders found in:\n  {param_folder}"
                )
            run_folder = os.path.join(param_folder, candidates[-1])
            logger.info(
                f"batch_id not specified — using most recent batch: {candidates[-1]}"
            )

        logger.info(f"Reconstructing users from: {run_folder}")

        # ------------------------------------------------------------------
        # Discover how many users are in this batch
        # ------------------------------------------------------------------
        user_indices = sorted(
            int(fname[len("user_"):-len("_history.parquet")])
            for fname in os.listdir(run_folder)
            if fname.startswith("user_") and fname.endswith("_history.parquet")
        )
        if not user_indices:
            raise RuntimeError(
                f"_reconstruct_users_from_raw: no user_*_history.parquet files in:\n  {run_folder}"
            )

        # ------------------------------------------------------------------
        # Restore simulation-level attributes needed by User objects
        # ------------------------------------------------------------------
        self.ro               = load_params["ro"]
        self.gamma            = load_params["gamma"]
        self.a_param          = load_params["a_param"]
        self.grid_exists      = grid_exists
        self._starting_position = starting_position

        if grid_exists and grid_params is not None:
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]

        self.users_list = []
        n_users   = len(user_indices)
        milestone = max(1, n_users // 10)

        history_header = [
            "State(E/R)", "Departure",
            "Departure_x_coord", "Departure_y_coord",
            "Destination", "Destination_x_coord", "Destination_y_coord",
            "Distance", "Angle", "Delta_t", "Clock",
        ]

        for pos, i in enumerate(user_indices):
            h_path = os.path.join(run_folder, f"user_{i}_history.parquet")
            v_path = os.path.join(run_folder, f"user_{i}_venue.parquet")
            g_path = os.path.join(run_folder, f"user_{i}_grid.parquet")

            history_df = pd.read_parquet(h_path)
            venue_df   = pd.read_parquet(v_path) if os.path.exists(v_path) else None

            # Rebuild history as list-of-lists (matching the live format)
            user_history  = [history_header] + history_df.values.tolist()

            venue_header  = list(venue_df.columns)
            venue_history = [venue_header] + venue_df.values.tolist()
            
            # Last row of history -> current state
            last = user_history[-1]
            col  = {name: idx for idx, name in enumerate(history_header)}

            last_clock    = last[col["Clock"]]
            last_position = (last[col["Destination_x_coord"]], last[col["Destination_y_coord"]])
            last_prev_pos = (last[col["Departure_x_coord"]],   last[col["Departure_y_coord"]])
            last_departure= last[col["Departure"]]
            last_state    = last[col["State(E/R)"]]

            # Rebuild _venue_index: venue_name -> row index in venue_history
            venue_index = {}
            for row_idx, row in enumerate(venue_history[1:], start=1):
                venue_index[row[0]] = row_idx

            # ------------------------------------------------------------------
            # Build a minimal User object without calling first_exploration()
            # We achieve this by calling __new__ and populating attributes manually
            # ------------------------------------------------------------------
            user = model.user.User.__new__(model.user.User)
            user.uid          = f"User_{i}"
            user.a_param      = load_params["a_param"]
            user.b_param      = load_params["b_param"]
            user.simulation   = self

            user.has_grid = grid_exists
            if grid_exists and os.path.exists(g_path):
                grid_df = pd.read_parquet(g_path)
                g = model.grid.Grid.__new__(model.grid.Grid)
                g.cell_lenght    = grid_params["cell_lenght"]
                g.cell_number    = grid_params["cell_number"]
                g.spatial_center = (g.cell_lenght * g.cell_number) / 2
                g.matrix         = grid_df.values.astype(int)
                user.grid = g
            elif grid_exists:
                logger.warning(f"User_{i}: grid.parquet not found, creating fresh grid.")
                user.grid = model.grid.Grid(
                    cell_lenght=grid_params["cell_lenght"],
                    cell_number=grid_params["cell_number"],
                )

            user.user_history       = user_history
            user.venue_history      = venue_history
            user._venue_index       = venue_index
            user.num_distinct_venues= len(venue_history) - 1

            user.clock              = last_clock
            user.position           = last_position
            user.previous_position  = last_prev_pos
            user.departure          = last_departure
            user.state              = last_state

            # Transient movement attributes (not needed to resume, but must exist)
            user.next_position      = None
            user.destination        = None
            user.previous_destination = None
            user.next_venue_return  = None
            user.delta_r            = None
            user.angle              = None
            user.delta_t            = None
            user.r_cm               = None
            user.r_gyr              = None
            user.stats              = [["metric", "value"]]
            user._row_index         = None

            self.users_list.append(user)

            if (pos + 1) % milestone == 0 or (pos + 1) == n_users:
                pct = ((pos + 1) / n_users) * 100
                logger.info(f"Reconstruction progress: {pct:.0f}% ({pos+1}/{n_users} users)")

        logger.info(
            f"Reconstructed {n_users} users from batch: {os.path.basename(run_folder)}"
        )

        # Store so that save_combined_raw_data() and run_analysis() can build
        # the two-level folder structure without the caller having to pass it again.
        self._load_params = load_params

        return run_folder

    def save_raw_data(self):
        """
        Serialise user_history and venue_history for every simulated user to
        Parquet files under raw_data/. When a grid is active, the grid matrix
        is also saved per user.

        Must be called after run().

        Folder layout
        -------------
        raw_data/
            ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                {YYYYMMDD_HHMMSS_ffffff}/
                    user_0_history.parquet
                    user_0_venue.parquet
                    [user_0_grid.parquet]   (only when grid is active)
                    ...

        Returns
        -------
        str
            Path to the timestamped run folder that was just created.
        """
        logger.info("Saving raw simulation data...")

        if self.grid_exists:
            run_folder = self._make_run_folder("raw_data", self._simulate_params, self._grid_params)
        else:
            run_folder = self._make_run_folder("raw_data", self._simulate_params)
        total     = len(self.users_list)
        milestone = max(1, total // 10)

        for i, user in enumerate(self.users_list):
            user_df  = pd.DataFrame(user.user_history[1:],  columns=user.user_history[0])
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])

            if user_df.empty:
                logger.warning(f"User_{i} (uid={user.uid}): user_history is empty — saving empty Parquet. "
                       f"This will cause errors on reconstruction. Consider increasing simulation time.")

            user_df.to_parquet( os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
            venue_df.to_parquet(os.path.join(run_folder, f"user_{i}_venue.parquet"),   index=False)

            if self.grid_exists:
                grid_df = pd.DataFrame(user.grid.matrix)
                grid_df.columns = grid_df.columns.astype(str)
                grid_df.to_parquet(os.path.join(run_folder, f"user_{i}_grid.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Save progress: {pct:.0f}% ({i+1}/{total} users saved)")

        logger.info(f"Raw data saved to: {run_folder}")
        return run_folder

    def save_combined_raw_data(self):
        """
        Save raw trajectories for a chained simulation (load → resume) under
        combined_raw_data/ using a two-level parameter folder structure that
        makes the provenance explicit.

        Folder layout
        -------------
        combined_raw_data/
            <base_params>/          ← parameters of the loaded batch (load_params)
                <resume_params>/    ← parameters of the resumed simulation (_simulate_params)
                    {YYYYMMDD_HHMMSS_ffffff}/
                        user_0_history.parquet
                        user_0_venue.parquet
                        [user_0_grid.parquet]

        Must be called after _reconstruct_users_from_raw() and resume().

        Returns
        -------
        str
            Path to the timestamped run folder that was just created.
        """
        logger.info("Saving combined raw simulation data...")

        if not hasattr(self, "_load_params") or self._load_params is None:
            raise RuntimeError(
                "save_combined_raw_data() requires _load_params to be set. "
                "Call _reconstruct_users_from_raw() before this method."
            )

        gp = getattr(self, "_grid_params", None)
        run_folder = self._make_combined_run_folder(
            "combined_raw_data",
            base_params   = self._load_params,
            resume_params = self._simulate_params,
            grid_params   = gp,
            batch_ts      = getattr(self, "_batch_ts", None),
        )

        total     = len(self.users_list)
        milestone = max(1, total // 10)

        for i, user in enumerate(self.users_list):
            user_df  = pd.DataFrame(user.user_history[1:],  columns=user.user_history[0])
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])

            user_df.to_parquet( os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
            venue_df.to_parquet(os.path.join(run_folder, f"user_{i}_venue.parquet"),   index=False)

            if self.grid_exists:
                grid_df = pd.DataFrame(user.grid.matrix)
                grid_df.columns = grid_df.columns.astype(str)
                grid_df.to_parquet(os.path.join(run_folder, f"user_{i}_grid.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Save progress: {pct:.0f}% ({i+1}/{total} users saved)")

        logger.info(f"Combined raw data saved to: {run_folder}")
        return run_folder

    # ==========================================================================
    # PHASE 2: ANALYSIS
    # ==========================================================================

    def run_analysis(self, analyse_params, raw_run_folder=None, grid_params=None, combined=False):
        """
        Compute all analysis quantities from raw trajectory data and save them
        to analysis_data/.

        Calling patterns
        ----------------
        (a) Standalone (ANALYSE mode)
            raw_run_folder=None -- data loaded from raw_data/ via analyse_params.
            grid_params must be supplied when the simulation was run with a grid,
            so that the correct raw_data folder is located and the analysis_data
            folder is named consistently.

        (b) Chained (SIMULATE_AND_ANALYSE mode)
            Pass raw_run_folder returned by save_raw_data(). Users already in
            memory are used directly without any disk read.
            grid_params is taken from self._grid_params when not explicitly
            provided (automatically set during run()).

        Files saved per user
        --------------------
        user_{i}_history.parquet      : full user trajectory with added columns:
                                        'rg'  — radius of gyration at each step
                                        's'   — cumulative distinct venues (exploration rows only)
        user_{i}_venue.parquet        : venue visit counts   (copy from raw)
        user_{i}_stats.parquet        : rg, rg_two, x_cm, y_cm

        Parameters
        ----------
        analyse_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time".
            Note: "user_number" is intentionally absent -- analysis always
            processes complete run folders, never a subset of users.
        raw_run_folder : str, optional
            Path to a specific raw_data run folder (simulate_and_analyse mode).
        grid_params : dict, optional
            Grid parameters dict with keys "cell_lenght" and "cell_number".
            Required when the simulation used a grid. In simulate_and_analyse
            mode this is inferred automatically from self._grid_params.

        Returns
        -------
        str
            Path to the timestamped analysis run folder that was just created.
        """
        logger.info("Starting analysis phase...")

        # In simulate_and_analyse mode, inherit grid_params from the run phase
        if grid_params is None and getattr(self, "_grid_params", None) is not None:
            grid_params = self._grid_params

        # Collect raw run folder paths to lock after analysis
        raw_folders_to_lock = set()

        # Load from disk only when there are no in-memory users
        if not self.users_list:
            #Directly callable. Main doesnt include this functionality
            if raw_run_folder is not None:
                logger.info(f"Loading raw data from: {raw_run_folder}")
                i = 0
                while True:
                    h_path = os.path.join(raw_run_folder, f"user_{i}_history.parquet")
                    if not os.path.exists(h_path):
                        break
                    v_path = os.path.join(raw_run_folder, f"user_{i}_venue.parquet")
                    uh = pd.read_parquet(h_path)
                    vh = pd.read_parquet(v_path) if os.path.exists(v_path) else None
                    self.users_list.append((uh, vh))
                    i += 1
                raw_folders_to_lock.add(raw_run_folder)
                logger.info(f"Loaded {i} users from raw run folder")
            else:
                #Analyse mode: load from raw_data/ using analyse_params filters
                logger.info("Loading raw data via analyse_params...")
                alg = analyse_params.get("algorithm", "ce")
                grid_active = alg != "ce"
                filters = {k: v for k, v in analyse_params.items()
                           if k not in ("algorithm", "starting_position")}
                if grid_active:
                    filters.update(grid_params) #Adding grid values
                loaded = model.utils.load_raw_parquet(
                    filters=filters,
                    grid=grid_active,
                    return_run_folders=True,
                )
                self.users_list = [entry["data"] for entry in loaded]
                raw_folders_to_lock.update(entry["run_folder"] for entry in loaded)
                logger.info(f"Loaded {len(self.users_list)} users from raw_data")
        elif raw_run_folder is not None:
            # simulate_and_analyse: users in memory, raw folder still known
            raw_folders_to_lock.add(raw_run_folder)

        #Logic for "lsa" mode
        if combined:
            if not hasattr(self, "_load_params") or self._load_params is None:
                raise RuntimeError(
                    "run_analysis(combined=True) requires _load_params to be set. "
                    "Call _reconstruct_users_from_raw() before this method."
                )
            batch_run_folder = self._make_combined_run_folder(
                "combined_analysis_data",
                base_params   = self._load_params,
                resume_params = analyse_params,
                grid_params   = grid_params,
                batch_ts      = getattr(self, "_batch_ts", None),
            )
            # Create the two variant subfolders inside the batch timestamp folder
            analysis_folder_mixed = os.path.join(batch_run_folder, "mixed")
            analysis_folder_pure  = os.path.join(batch_run_folder, "pure")
            os.makedirs(analysis_folder_mixed, exist_ok=True)
            os.makedirs(analysis_folder_pure,  exist_ok=True)
            # The clock threshold that separates the "loaded" phase from the "resumed" phase
            _pure_t_start = self._load_params["time"]
        else:
            alg_for_folder = analyse_params.get("algorithm", getattr(self, "_simulate_params", {}).get("algorithm", "ce"))
            grid_active    = alg_for_folder != "ce"
            if grid_active:
                analysis_folder = self._make_run_folder("analysis_data", analyse_params, grid_params)
            else:
                analysis_folder = self._make_run_folder("analysis_data", analyse_params)

        total     = len(self.users_list)
        milestone = max(1, total // 10)

        # Determine which folders to write into.
        # In combined (lsa) mode we write two variants:
        #   mixed/ → full history (loaded phase + resumed phase), same as before
        #   pure/  → only the resumed phase (rows with Clock > _pure_t_start)
        # In non-combined mode we use a single analysis_folder as before.
        if combined:
            _variant_folders = [
                ("mixed", analysis_folder_mixed),
                ("pure",  analysis_folder_pure),
            ]
        else:
            _variant_folders = [
                (None, analysis_folder),
            ]

        # Collect full DataFrames for Gini / batch-stats (always computed on "mixed" / full data)
        user_history_df_list  = []
        venue_history_df_list = []

        # We iterate users once, build both variants in the same pass.
        for i, user_entry in enumerate(self.users_list):
            if isinstance(user_entry, model.user.User):
                user_history_df  = pd.DataFrame(
                    user_entry.user_history[1:],  columns=user_entry.user_history[0])
                venue_history_df = pd.DataFrame(
                    user_entry.venue_history[1:], columns=user_entry.venue_history[0])
            elif isinstance(user_entry, dict):
                user_history_df  = user_entry["user_history"]
                venue_history_df = user_entry["venue_history"]
            else:
                user_history_df, venue_history_df = user_entry

            user_history_df_list.append(user_history_df)
            venue_history_df_list.append(venue_history_df)

            # Resolve grid_df once (shared across variants)
            grid_df = None
            alg_key = analyse_params.get("algorithm", getattr(self, "_simulate_params", {}).get("algorithm", "ce"))
            if alg_key != "ce":
                if isinstance(user_entry, model.user.User) and user_entry.has_grid:
                    grid_df = pd.DataFrame(user_entry.grid.matrix)
                elif isinstance(user_entry, dict) and user_entry.get("grid") is not None:
                    grid_df = user_entry["grid"]
                if grid_df is not None:
                    grid_df.columns = grid_df.columns.astype(str)

            for _variant, _folder in _variant_folders:
                # For "pure" variant: trim history to rows after the loaded phase clock
                if _variant == "pure":
                    uh = user_history_df[user_history_df["Clock"] > _pure_t_start].copy()
                    # Rebuild venue_history counting only visits that appear in the pure window
                    pure_venues = uh["Destination"].dropna().unique()
                    vh = venue_history_df[venue_history_df.iloc[:, 0].isin(pure_venues)].copy()
                else:
                    uh = user_history_df
                    vh = venue_history_df

                stats_df        = model.utils.compute_stats(uh, vh)
                rg_over_time_df = model.utils.compute_rg_over_time(uh)
                s_over_time_df  = model.utils.compute_s_over_time(uh)

                dist_col  = uh["Distance"].dropna()
                dist_col  = dist_col[dist_col > 0]
                mean_dist = float(dist_col.mean()) if len(dist_col) > 0 else 0.0
                num_venues = int(len(vh))

                extra_rows = pd.DataFrame([
                    {"metric": "mean_travel_distance", "value": mean_dist},
                    {"metric": "num_venues",            "value": float(num_venues)},
                ])
                stats_df = pd.concat([stats_df, extra_rows], ignore_index=True)

                # Merge rg_over_time and s_over_time into user_history as extra columns.
                # rg_over_time has one row per step (same index as uh); s_over_time has
                # one row per exploration event only (rows where State(E/R) == "E").
                # We align both on the Clock value so NaN fills non-exploration rows for 's'.
                uh = uh.copy()
                # rg column: one value per history row (same length as uh)
                uh["rg"] = rg_over_time_df["rg"].values

                # s column: only present for exploration rows — merge on clock, then
                # forward-fill so return rows carry the last known s value instead of NaN
                s_map = dict(zip(s_over_time_df["clock"].values, s_over_time_df["s"].values))
                uh["s"] = uh["Clock"].map(s_map).ffill()

                uh.to_parquet(
                    os.path.join(_folder, f"user_{i}_history.parquet"),      index=False)
                vh.to_parquet(
                    os.path.join(_folder, f"user_{i}_venue.parquet"),        index=False)
                stats_df.to_parquet(
                    os.path.join(_folder, f"user_{i}_stats.parquet"),        index=False)

                if grid_df is not None:
                    grid_df.to_parquet(
                        os.path.join(_folder, f"user_{i}_grid.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Analysis progress: {pct:.0f}% ({i+1}/{total} users)")

        # ------------------------------------------------------------------
        # Batch-level summary stats and aggregate grids.
        # In combined mode: compute for each variant folder independently.
        # In non-combined mode: compute once for the single analysis_folder.
        # ------------------------------------------------------------------
        def _compute_and_save_batch_stats(target_folder, uhdf_list, vhdf_list):
            gini_ind  = model.utils.compute_individual_gini(vhdf_list)
            gini_coll = model.utils.compute_collective_gini(vhdf_list)

            _all_rg      = []
            _all_2rg     = []
            _all_md      = []
            _all_nv      = []
            _all_s_rand  = []
            _all_s_unc   = []
            _all_s_real  = []
            for j in range(len(uhdf_list)):
                _sf = pd.read_parquet(os.path.join(target_folder, f"user_{j}_stats.parquet"))
                def _get(metric, _sf=_sf):
                    row = _sf.loc[_sf["metric"] == metric, "value"]
                    return float(row.iloc[0]) if len(row) else float("nan")
                _all_rg.append(     _get("radius of gyration"))
                _all_2rg.append(    _get("radius of gyration two"))
                _all_md.append(     _get("mean_travel_distance"))
                _all_nv.append(     _get("num_venues"))
                _all_s_rand.append( _get("s_rand"))
                _all_s_unc.append(  _get("s_unc"))
                _all_s_real.append( _get("s_real"))

            bsdf = pd.DataFrame([
                {"metric": "gini_collective",      "value": gini_coll},
                {"metric": "gini_individual",      "value": gini_ind},
                {"metric": "n_users",              "value": float(total)},
                {"metric": "mean_rg",              "value": float(np.nanmean(_all_rg))},
                {"metric": "mean_2rg",             "value": float(np.nanmean(_all_2rg))},
                {"metric": "mean_travel_distance", "value": float(np.nanmean(_all_md))},
                {"metric": "mean_num_venues",      "value": float(np.nanmean(_all_nv))},
                {"metric": "mean_s_rand",          "value": float(np.nanmean(_all_s_rand))},
                {"metric": "mean_s_unc",           "value": float(np.nanmean(_all_s_unc))},
                {"metric": "mean_s_real",          "value": float(np.nanmean(_all_s_real))},
            ])

            grid_active = analyse_params.get("algorithm", getattr(self, "_simulate_params", {}).get("algorithm", "ce")) != "ce"
            if grid_active:
                batch_grid, _ = model.utils.compute_and_save_aggregate_grid(target_folder)
                moran_i_dict  = model.utils.compute_moran_i(batch_grid)
                moran_df      = pd.DataFrame([{"metric": k, "value": v} for k, v in moran_i_dict.items()])
                bsdf = pd.concat([bsdf, moran_df], ignore_index=True)

            bsdf.to_parquet(os.path.join(target_folder, "batch_stats.parquet"), index=False)
            logger.info(f"Analysis data saved to: {target_folder}")

        if combined:
            # mixed: full DataFrames
            _compute_and_save_batch_stats(analysis_folder_mixed, user_history_df_list, venue_history_df_list)
            # pure: rebuild trimmed lists from the saved parquets (already on disk)
            pure_uhdf_list = []
            pure_vhdf_list = []
            for j in range(total):
                pure_uhdf_list.append(
                    pd.read_parquet(os.path.join(analysis_folder_pure, f"user_{j}_history.parquet")))
                pure_vhdf_list.append(
                    pd.read_parquet(os.path.join(analysis_folder_pure, f"user_{j}_venue.parquet")))
            _compute_and_save_batch_stats(analysis_folder_pure, pure_uhdf_list, pure_vhdf_list)
            # The batch run folder returned to the caller is the parent timestamp folder
            analysis_folder = batch_run_folder
        else:
            _compute_and_save_batch_stats(analysis_folder, user_history_df_list, venue_history_df_list)

        for folder in raw_folders_to_lock:
            lock_path = os.path.join(folder, "_analyzed.lock")
            with open(lock_path, "w") as f:
                f.write(f"Analyzed -> {analysis_folder}\n")
            logger.info(f"Locked raw folder: {folder}")

        return analysis_folder

    # ==========================================================================
    # PHASE 3: PLOT
    # ==========================================================================

    def load_analysis_data(self, plot_params, grid_params=None, plot_list=None):
        """
        Load pre-computed analysis data from analysis_data/ into memory.

        Parameters
        ----------
        plot_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time".
            Optional: "user_number" (positive = first N, negative = last N, absent = all),
            "algorithm" (str), "starting_position" (str).
        grid_params : dict, optional
            When provided (and algorithm implies a grid), must contain
            "cell_lenght" and "cell_number". Merged into the filter dict passed
            to the loader so the correct analysis folder is located.
        plot_list : list of str, optional
            The same plot_list that will be passed to plot(). When provided,
            only the files required for those plots are loaded from disk,
            and aggregate grids are loaded only when needed.
            When None (default), all files are loaded (backward-compatible).
        """
        # ------------------------------------------------------------------
        # Determine which per-user files and which aggregate data are needed
        # ------------------------------------------------------------------
        _PLOT_FILE_NEEDS = {
            "mobility":           {"user_history", "venue_history", "stats"},
            "rg_distribution":    {"stats"},
            "dist_distribution":  {"user_history"},
            "freq_distribution":  {"venue_history"},
            "two_radius":         {"stats"},
            "s_over_time":        {"user_history"},
            "rg_over_time":       {"user_history"},
            "aggregate_grid":     set(),
            "batch_grid":         set(),
            "batch_stats":        set(),
            "srand":              {"stats"},
            "sunc":               {"stats"},
            "sreal":              {"stats"},
            "s_triple":           {"stats"},
            "venue_visits":       {"venue_history"},
            "venue_popularity":   {"venue_history"},
        }

        if plot_list is not None:
            needed_files = set()
            for p in plot_list:
                needed_files |= _PLOT_FILE_NEEDS.get(p, set())
            needs_user_data   = bool(needed_files)
            needs_grids       = any(p in ("aggregate_grid", "batch_grid") for p in plot_list)
            needs_batch_stats = "batch_stats" in plot_list
        else:
            needed_files      = None   # load everything
            needs_user_data   = True
            needs_grids       = True
            needs_batch_stats = True

        logger.info("Loading analysis data...")
        if plot_list is not None:
            logger.info(f"Plots requested: {plot_list}")
            if needs_user_data:
                logger.info(f"Per-user files needed: {sorted(needed_files)}")
            else:
                logger.info("No per-user files required — skipping per-user load")
            if needs_grids:
                logger.info("Aggregate grids will be loaded")
            else:
                logger.info("No grid plots requested — skipping aggregate grid load")

        combined = plot_params.get("combined", False)

        # Derive whether a grid was used from the algorithm field.
        # For combined params (cp/mp modes) the key is "algorithm_0" (the base run);
        # for normal params (p mode) the key is "algorithm".
        if combined:
            alg0        = plot_params.get("algorithm_0", "ce")
            grid_active = alg0 != "ce"
        else:
            alg         = plot_params.get("algorithm", "ce")
            grid_active = alg != "ce"

        filters = {**plot_params}
        if grid_active and grid_params:
            filters.update(grid_params)

        # Resolve variant ("p" → "pure", "m" → "mixed", default "mixed").
        # Only meaningful in combined (cp/mp) mode; ignored in plain p mode.
        _variant_raw = plot_params.get("variant", "m")
        variant = "pure" if _variant_raw == "p" else "mixed"
        if combined:
            filters["variant"] = variant

        # gamma needed by freq_distribution plot
        self.gamma = plot_params.get("gamma")

        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []
        self.all_s_over_time = []
        self.mean_dist_list = []
        self.s_rand_list = []
        self.s_unc_list  = []
        self.s_real_list = []

        # ------------------------------------------------------------------
        # Per-user data load (skipped when only grid plots are requested)
        # ------------------------------------------------------------------
        loaded = []
        if needs_user_data:
            if combined:
                loaded = model.utils.load_combined_analysis_parquet(filters=filters, needed_files=needed_files)
            else:
                loaded = model.utils.load_analysis_parquet(filters=filters, needed_files=needed_files)

        # Keep a copy of the full filters with the actual user count for titles/file names.
        # _plot_grid_params excludes user_number (irrelevant for grid/aggregate plots).
        self._plot_params = {**filters, "user_number": len(loaded)}
        self._plot_grid_params = {k: v for k, v in filters.items() if k != "user_number"}
        self._grid_active = grid_active  # saved for use in plot()

        for entry in loaded:
            stats_df         = entry.get("stats")
            venue_history_df = entry.get("venue_history")
            user_history_df  = entry.get("user_history")

            rg_over_time_df = None
            s_over_time_df  = None
            if user_history_df is not None:
                if "rg" in user_history_df.columns:
                    rg_over_time_df = user_history_df[["Clock", "rg"]].rename(
                        columns={"Clock": "clock"})
                if "s" in user_history_df.columns:
                    s_rows = user_history_df[["Clock", "s"]].dropna(subset=["s"])
                    s_over_time_df = s_rows.rename(columns={"Clock": "clock"})

            self.last_user_history  = user_history_df
            self.last_venue_history = venue_history_df
            self.last_stats         = stats_df

            if stats_df is not None:
                def _sget(m):
                    row = stats_df.loc[stats_df['metric'] == m, 'value']
                    return float(row.iloc[0]) if len(row) else float("nan")

                self.rg_list.append(    _sget("radius of gyration"))
                self.two_rg_list.append(_sget("radius of gyration two"))
                self.last_x_cm = _sget("x_center of mass")
                self.last_y_cm = _sget("y_center of mass")
                md = _sget("mean_travel_distance")
                if not np.isnan(md):
                    pass  # mean_travel_distance retained in stats for other uses

                self.s_rand_list.append(_sget("s_rand"))
                self.s_unc_list.append( _sget("s_unc"))
                self.s_real_list.append(_sget("s_real"))

            # -- collect all individual delta_r values for dist_distribution --
            if user_history_df is not None:
                dist_col = user_history_df["Distance"].dropna()
                dist_col = dist_col[dist_col > 0]
                self.mean_dist_list.extend(dist_col.tolist())

            if venue_history_df is not None:
                self.venue_history_list.append(venue_history_df)
            if rg_over_time_df is not None:
                self.all_rg_over_time.append(rg_over_time_df)
            if s_over_time_df is not None:
                self.all_s_over_time.append(s_over_time_df)
        #self.last_user_history.to_csv(r"C:\Users\Jacopo\Desktop\nome_file.csv", index=False)

        # ------------------------------------------------------------------
        # Aggregate grid load (skipped when no grid plots are requested)
        # ------------------------------------------------------------------
        if needs_grids:
            if combined:
                self.aggregate_grid, self.aggregate_grid_user_count, self.batch_grids = \
                    model.utils.load_combined_all_aggregate_grids(filters)
            else:
                self.aggregate_grid, self.aggregate_grid_user_count, self.batch_grids = \
                    model.utils.load_all_aggregate_grids(filters)
        else:
            self.aggregate_grid            = None
            self.aggregate_grid_user_count = 0
            self.batch_grids               = []

        # ------------------------------------------------------------------
        # Batch stats load (skipped when batch_stats is not in plot_list)
        # ------------------------------------------------------------------
        if needs_batch_stats:
            if combined:
                self.batch_stats_list = model.utils.load_combined_all_batch_stats(filters)
            else:
                self.batch_stats_list = model.utils.load_all_batch_stats(filters)
        else:
            self.batch_stats_list = []

    def plot(self, plot_list, show=False):
        """
        Generate all plots listed in plot_list.

        Must be called after load_analysis_data().

        Parameters
        ----------
        plot_list : list of str
            Accepted values: "mobility", "rg_distribution", "freq_distribution",
            "two_radius", "s_over_time", "rg_over_time".
        """
        self._show = show
        p  = self._plot_params
        pg = self._plot_grid_params

        if "mobility" in plot_list:
            model.utils_plot.plot_mobility_visualization(
                self.last_user_history, self.last_venue_history,
                stats=self.last_stats,
                filters=p, title=model.utils_plot.build_plot_title("User Mobility", p),
                show=self._show
            )
        if "rg_distribution" in plot_list:
            model.utils_plot.plot_rg_distribution(
                self.rg_list,
                filters=p, title=model.utils_plot.build_plot_title("RG Distribution", p),
                show=self._show
            )
        if "dist_distribution" in plot_list:
            model.utils_plot.plot_dist_distribution(
                self.mean_dist_list,
                filters=p, title=model.utils_plot.build_plot_title("Travel Distance Distribution", p),
                show=self._show
            )
        if "freq_distribution" in plot_list:
            model.utils_plot.plot_frequency_distribution(
                self.venue_history_list, self.gamma,
                filters=p, title=model.utils_plot.build_plot_title("Frequency Distribution", p),
                show=self._show
            )
        if "two_radius" in plot_list:
            model.utils_plot.plot_two_radius_vs_radius(
                self.rg_list, self.two_rg_list,
                filters=p, title=model.utils_plot.build_plot_title("Two-RG vs RG", p),
                show=self._show
            )
        if "s_over_time" in plot_list:
            model.utils_plot.plot_S_over_time_from_df(
                self.all_s_over_time, filters=p, bins_per_decade=10,
                title=model.utils_plot.build_plot_title("S over time", p),
                show=self._show
            )
        if "rg_over_time" in plot_list:
            model.utils_plot.plot_r_gyr_over_time_from_df(
                self.all_rg_over_time, filters=p, bins_per_decade=10,
                title=model.utils_plot.build_plot_title("Radius of Gyration over time", p),
                show=self._show
            )
        if "aggregate_grid" in plot_list:
            if self.aggregate_grid is not None:
                pg_with_n = {**pg, "u_n": self.aggregate_grid_user_count}
                model.utils_plot.plot_aggregate_grid(
                    self.aggregate_grid, filters=pg_with_n,
                    title=model.utils_plot.build_plot_title("Aggregate Grid", pg_with_n),
                    mask_center= self._grid_active and p.get("starting_position", "centered") == "centered",
                    show=self._show
                )
            else:
                logger.warning("No param-level aggregate grid available to plot.")
        if "batch_grid" in plot_list:
            if self.batch_grids:
                for batch_id, grid, batch_user_count in self.batch_grids:
                    pg_with_n = {**pg, "u_n": batch_user_count}
                    model.utils_plot.plot_aggregate_grid(
                        grid, filters=pg_with_n,
                        title=model.utils_plot.build_plot_title(f"Grid - {batch_id}", pg_with_n),
                        batch_id=batch_id,
                        mask_center= self._grid_active and p.get("starting_position", "centered") == "centered",
                        show=self._show
                    )
            else:
                logger.warning("No batch-level aggregate grids available to plot.")
        if "batch_stats" in plot_list:
            if self.batch_stats_list:
                model.utils_plot.print_batch_stats(
                    self.batch_stats_list,
                    filters=pg,
                    title=model.utils_plot.build_plot_title("Batch Stats", pg),
                )
            else:
                logger.warning("No batch stats available to print.")

        if "srand" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                self.s_rand_list, entropy_type="srand",
                filters=p, title=model.utils_plot.build_plot_title("S^rand Distribution", p),
                show=self._show,
            )
        if "sunc" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                self.s_unc_list, entropy_type="sunc",
                filters=p, title=model.utils_plot.build_plot_title("S^unc Distribution", p),
                show=self._show,
            )
        if "sreal" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                self.s_real_list, entropy_type="sreal",
                filters=p, title=model.utils_plot.build_plot_title("S^real Distribution", p),
                show=self._show,
            )
        if "s_triple" in plot_list:
            model.utils_plot.plot_entropy_triple(
                self.s_rand_list, self.s_unc_list, self.s_real_list,
                filters=p, title=model.utils_plot.build_plot_title("Entropy Distributions", p),
                show=self._show,
            )
        if "venue_visits" in plot_list:
            model.utils_plot.plot_venue_ranking(
                self.venue_history_list, mode="visits",
                filters=p, title=model.utils_plot.build_plot_title("Venue Visits Ranking", p),
                show=self._show,
            )
        if "venue_popularity" in plot_list:
            model.utils_plot.plot_venue_ranking(
                self.venue_history_list, mode="popularity",
                filters=p, title=model.utils_plot.build_plot_title("Venue Popularity Ranking", p),
                show=self._show,
            )

    # ==========================================================================
    # MULTIPLEPLOT  ("mp" mode)
    # ==========================================================================

    @staticmethod
    def _expand_multiple_plot_params(mp_params):
        """
        Expand a multiple_plot_params dict that may contain list/tuple values
        into a list of plain dicts (one per combination) and a matching list
        of label strings.

        Rules
        -----
        * The key ``"label"`` must be a list/tuple of strings — one per
          expected combination.  It is popped before building combinations.
        * The key ``"combined"`` is scalar and is copied verbatim to every
          expansion.
        * Every other key whose value is a list/tuple participates in the
          cartesian product.  Scalar values are shared across all combinations.
        * The number of label strings must equal the number of combinations
          produced by the cartesian product; a ValueError is raised otherwise.

        Returns
        -------
        list of (label_str, params_dict)
        """
        import itertools

        params = dict(mp_params)
        labels = params.pop("label")
        if isinstance(labels, str):
            labels = [labels]

        # Separate list/scalar values (skip "combined" — always scalar)
        list_keys   = [k for k, v in params.items()
                       if isinstance(v, (list, tuple)) and k != "combined"]
        scalar_keys = [k for k in params if k not in list_keys]

        if not list_keys:
            # No expansion needed — single combination
            return [(labels[0] if labels else "", params)]

        # Build cartesian product of all list values
        value_combinations = list(itertools.product(*[params[k] for k in list_keys]))
        scalar_part = {k: params[k] for k in scalar_keys}

        if len(labels) != len(value_combinations):
            raise ValueError(
                f"multiple_plot_params: 'label' has {len(labels)} entries but "
                f"the cartesian product of list values yields "
                f"{len(value_combinations)} combinations.\n"
                f"List keys: {list_keys}"
            )

        result = []
        for label, combo in zip(labels, value_combinations):
            combo_dict = {**scalar_part, **dict(zip(list_keys, combo))}
            result.append((label, combo_dict))
        return result

    def load_multiple_plot_data(self, mp_params, grid_params=None, plot_list=None):
        """
        Load one analysis dataset per combination encoded in *mp_params* and
        store each as a named series in self._mp_series for use by
        multipleplot().

        Parameters
        ----------
        mp_params : dict
            Like combined_plot_params but with:
            - ``"label"``  : list of strings, one per series
            - List/tuple values on any other key trigger cartesian expansion.
            - ``"combined": True`` must be present (mp works on combined data).
        grid_params : dict, optional
        plot_list : list of str, optional
        """
        self._mp_series = []
        expansions = self._expand_multiple_plot_params(mp_params)
        logger.info(f"multipleplot: loading {len(expansions)} series")

        for label, p_dict in expansions:
            logger.info(f"  loading series '{label}' with params: {p_dict}")
            # Reuse the standard load machinery by calling load_analysis_data
            # on a throw-away Simulation object so we don't clobber self's data.
            tmp = Simulation()
            tmp.load_analysis_data(p_dict, grid_params=grid_params, plot_list=plot_list)

            series = {
                "label":               label,
                "rg_list":             list(tmp.rg_list),
                "two_rg_list":         list(tmp.two_rg_list),
                "mean_dist_list":      list(tmp.mean_dist_list),
                "all_rg_over_time":    list(tmp.all_rg_over_time),
                "all_s_over_time":     list(tmp.all_s_over_time),
                "batch_stats_list":    list(tmp.batch_stats_list),
                "_plot_params":        dict(tmp._plot_params),
                "s_rand_list":         list(tmp.s_rand_list),
                "s_unc_list":          list(tmp.s_unc_list),
                "s_real_list":         list(tmp.s_real_list),
                "venue_history_list":  list(tmp.venue_history_list),
            }
            self._mp_series.append(series)
            logger.info(
                f"  series '{label}': {len(series['rg_list'])} users, "
                f"{len(series['all_rg_over_time'])} rg_over_time records"
            )

    def multipleplot(self, plot_list, show=False):
        """
        Generate multi-series overlay plots for every entry in plot_list that
        supports it.

        Supported plots (overlaid on shared axes):
            "rg_distribution"   — log-log density of rg values
            "dist_distribution" — log-log density of mean travel distances
            "rg_over_time"      — normalised rg(t) / rg_final [%]
            "s_over_time"       — S(t) over time
            "two_radius"        — 2-rg vs rg scatter
            "freq_distribution" — rank-frequency (requires gamma; shared across series)

        Unsupported plots (spatial / single-user):
            "mobility", "aggregate_grid", "batch_grid"
            → silently skipped with a warning.

        "batch_stats" is printed once per series (each series gets its own table).
        """
        if not self._mp_series:
            logger.warning("multipleplot() called but _mp_series is empty. "
                           "Call load_multiple_plot_data() first.")
            return

        # Build a combined-filters dict from the first series for titles/saving
        first_p = self._mp_series[0]["_plot_params"]

        _UNSUPPORTED = {"mobility", "aggregate_grid", "batch_grid"}
        for pl in plot_list:
            if pl in _UNSUPPORTED:
                logger.warning(f"multipleplot: '{pl}' is not supported in mp mode — skipped.")

        if "rg_distribution" in plot_list:
            model.utils_plot.plot_rg_distribution(
                rg_list=None,
                series_list=[(s["label"], s["rg_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("RG Distribution (multi)", first_p),
                show=show,
            )

        if "dist_distribution" in plot_list:
            model.utils_plot.plot_dist_distribution(
                mean_dist_list=None,
                series_list=[(s["label"], s["mean_dist_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("Travel Distance Distribution (multi)", first_p),
                show=show,
            )

        if "rg_over_time" in plot_list:
            model.utils_plot.plot_r_gyr_over_time_from_df(
                rg_over_time_list=None,
                series_list=[(s["label"], s["all_rg_over_time"]) for s in self._mp_series],
                filters=first_p, bins_per_decade=10,
                title=model.utils_plot.build_plot_title("RG over time (multi)", first_p),
                show=show,
            )

        if "s_over_time" in plot_list:
            model.utils_plot.plot_S_over_time_from_df(
                s_over_time_list=None,
                series_list=[(s["label"], s["all_s_over_time"]) for s in self._mp_series],
                filters=first_p, bins_per_decade=10,
                title=model.utils_plot.build_plot_title("S over time (multi)", first_p),
                show=show,
            )

        if "two_radius" in plot_list:
            model.utils_plot.plot_two_radius_vs_radius(
                rg_list=None, two_rg_list=None,
                series_list=[(s["label"], s["rg_list"], s["two_rg_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("Two-RG vs RG (multi)", first_p),
                show=show,
            )

        if "batch_stats" in plot_list:
            for s in self._mp_series:
                if s["batch_stats_list"]:
                    model.utils_plot.print_batch_stats(
                        s["batch_stats_list"],
                        filters=s["_plot_params"],
                        title=model.utils_plot.build_plot_title(
                            f"Batch Stats [{s['label']}]", s["_plot_params"]),
                    )

        if "srand" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                entropy_list=None, entropy_type="srand",
                series_list=[(s["label"], s["s_rand_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("S^rand Distribution (multi)", first_p),
                show=show,
            )
        if "sunc" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                entropy_list=None, entropy_type="sunc",
                series_list=[(s["label"], s["s_unc_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("S^unc Distribution (multi)", first_p),
                show=show,
            )
        if "sreal" in plot_list:
            model.utils_plot.plot_entropy_distribution(
                entropy_list=None, entropy_type="sreal",
                series_list=[(s["label"], s["s_real_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("S^real Distribution (multi)", first_p),
                show=show,
            )

        if "venue_visits" in plot_list:
            model.utils_plot.plot_venue_ranking(
                venue_history_list=None, mode="visits",
                series_list=[(s["label"], s["venue_history_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("Venue Visits Ranking (multi)", first_p),
                show=show,
            )

        if "venue_popularity" in plot_list:
            model.utils_plot.plot_venue_ranking(
                venue_history_list=None, mode="popularity",
                series_list=[(s["label"], s["venue_history_list"]) for s in self._mp_series],
                filters=first_p,
                title=model.utils_plot.build_plot_title("Venue Popularity Ranking (multi)", first_p),
                show=show,
            )

    # ==========================================================================
    # UTILITY
    # ==========================================================================

    def _make_run_folder(self, data_type, params, grid_params=None):
        """
        Create and return a timestamped run folder under a data_type root.

        Folder hierarchy:
            <project_root>/<data_type>/
                ro_{ro}_g_{gamma}_a_{a}_b_{b}_t_{time}_[cl_{cl}_cn_{cn}]_sp_{sp}_alg_{algorithm}/
                    {YYYYMMDD_HHMMSS_ffffff}/

        Parameters
        ----------
        data_type : str
            e.g. "raw_data" or "analysis_data".
        params : dict
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
            Optional: "starting_position" (str, default "centered").
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, data_type)

        sp  = params.get("starting_position", "centered")
        alg = params.get("algorithm", "ce")

        # Map starting_position to single-letter abbreviation: c / r / n
        _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")

        if grid_params is None:
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_g_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_t_{params['time']}"
                f"_sp_{_sp_abbr}"
                f"_alg_{alg}"
            )
        else:
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_g_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_t_{params['time']}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_sp_{_sp_abbr}"
                f"_alg_{alg}"
            )
        run_folder = os.path.join(param_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def _make_combined_run_folder(self, data_type, base_params, resume_params, grid_params=None, batch_ts=None):
        """
        Create and return a timestamped run folder under a two-level parameter
        hierarchy for chained (load → resume) simulations.

        Folder hierarchy
        ----------------
        <project_root>/<data_type>/
            ro_{ro}_g_{gamma}_a_{a}_b_{b}_t0_{t0}_t1_{t1}
            [_cl_{cl}_cn_{cn}]_sp_{sp}_alg0_{alg0}_alg1_{alg1}/
                {YYYYMMDD_HHMMSS_ffffff}/

        Parameters
        ----------
        data_type : str
            Root directory name, e.g. "combined_raw_data" or
            "combined_analysis_data".
        base_params : dict
            Parameters of the original loaded batch.
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
        resume_params : dict
            Parameters used for the resume phase.
            Must contain "ro", "gamma", "a_param", "b_param", "time", "algorithm".
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".
        batch_ts : str, optional
            Timestamp string in the format "YYYYMMDD_HHMMSS_ffffff".
            When provided, this exact string is used as the batch folder name
            so that combined_raw_data, combined_analysis_data and the check
            folder all share the same ID.  When None, a fresh timestamp is
            generated.

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, data_type)

        ro      = base_params["ro"]
        gamma   = base_params["gamma"]
        a_param = base_params["a_param"]
        b_param = base_params["b_param"]
        sp      = base_params.get("starting_position", "centered")

        t0   = base_params["time"]
        t1   = resume_params["time"]
        alg0 = base_params.get("algorithm", "ce")
        alg1 = resume_params.get("algorithm", "ce")

        _sp_abbr = {"centered": "c", "random": "r", "normal": "n"}.get(sp, "c")

        if grid_params is not None:
            param_folder_name = (
                f"ro_{ro}_g_{gamma}"
                f"_a_{a_param}_b_{b_param}"
                f"_t0_{t0}_t1_{t1}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_sp_{_sp_abbr}"
                f"_alg0_{alg0}_alg1_{alg1}"
            )
        else:
            param_folder_name = (
                f"ro_{ro}_g_{gamma}"
                f"_a_{a_param}_b_{b_param}"
                f"_t0_{t0}_t1_{t1}"
                f"_sp_{_sp_abbr}"
                f"_alg0_{alg0}_alg1_{alg1}"
            )

        if batch_ts is None:
            batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_folder = os.path.join(
            base_folder,
            param_folder_name,
            batch_ts,
        )
        os.makedirs(run_folder, exist_ok=True)
        return run_folder