import numpy as np
import model.user
import model.utils
import model.utils_plot
import model.utils_user_knn
import model.grid
import pandas as pd
import os
from datetime import datetime
from pprint import pprint

import logging
logger = logging.getLogger(__name__)


class Simulation:
    """
    three-phase mobility simulation pipeline:

    Phase 1 - SIMULATE : run the EPR model for N users, save raw trajectories to raw_data/
    Phase 2 - ANALYSIS : load raw trajectories, compute analysis quantities, save to analysis_data/
    Phase 3 - PLOT     : load analysis data and generate all requested plots

    Each phase is self-contained and receives only the parameters it needs:
        run(simulate_params, grid_params)   requires ro, gamma, a_param, b_param, user_number, time
                                            optional: epr_fraction (default 1.0), top_k (default 10)
                                            grid_params optional: cell_lenght, cell_number
        run_analysis(analyse_params, grid_params)
                                            requires ro, gamma, a_param, b_param, time
                                            grid_params optional: cell_lenght, cell_number
        load_analysis_data(plot_params)     requires ro, gamma, a_param, b_param, time, [user_number]
        plot(plot_list)

    Additional entry points for chained simulations:
        resume(simulate_params, grid_params)
            Continue a simulation on users already loaded in memory (self.users_list must be
            populated either by run() or by _reconstruct_users_from_raw()).
            simulate_params["time"] is interpreted as the *new* absolute clock target: users
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
        Run the EPR (+ optional CF) simulation and store results in self.users_list.

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "user_number", "time".
            Optional:
              "epr_fraction" (float, default 1.0) -- fraction of total time simulated
                  with pure EPR before switching to collaborative filtering.
                  Must be > 0. When 1.0 (default) the simulation is pure EPR.
              "top_k" (int, default 10) -- number of most-similar neighbors considered
                  per user during the CF phase.
        """
        ro           = simulate_params["ro"]
        gamma        = simulate_params["gamma"]
        a_param      = simulate_params["a_param"]
        b_param      = simulate_params["b_param"]
        user_number  = simulate_params["user_number"]
        time         = simulate_params["time"]
        grid_exists  = simulate_params["grid"]
        random_start = simulate_params.get("random_start", False)
        epr_fraction = simulate_params.get("epr_fraction", 1.0)
        top_k        = simulate_params.get("top_k", 10)

        if epr_fraction <= 0:
            raise ValueError(
                "epr_fraction must be > 0. At least some EPR data is required "
                "before the CF phase can build user similarities."
            )

        # Store on self so User objects can access ro/gamma via self.simulation,
        # and save_raw_data() can build the correct folder name
        self.ro               = ro
        self.gamma            = gamma
        self.a_param          = a_param
        self.b_param          = b_param
        self._simulate_params = simulate_params
        self.grid_exists      = grid_exists
        self._random_start    = random_start

        if grid_exists == True:
            self._grid_params = grid_params
            self.grid_cell_lenght = grid_params["cell_lenght"]
            self.grid_cell_number = grid_params["cell_number"]
            if random_start:
                self._spatial_center = (self.grid_cell_lenght * self.grid_cell_number) / 2

        epr_time  = epr_fraction * time
        milestone = max(1, user_number // 10)
        logger.info(
            f"Starting EPR phase for {user_number} users "
            f"(epr_fraction={epr_fraction}, epr_time={epr_time})..."
        )

        # ---------------------------------------------------------------
        # Phase 1: EPR
        # ---------------------------------------------------------------
        for i in range(user_number):
            if grid_exists and random_start:
                sc = self._spatial_center
                init_pos = (
                    np.random.uniform(-sc, sc),
                    np.random.uniform(-sc, sc),
                )
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
            while user.clock < epr_time:
                user.make_decision()

            self.users_list.append(user)

            if (i + 1) % milestone == 0 or (i + 1) == user_number:
                pct = ((i + 1) / user_number) * 100
                logger.info(f"EPR progress: {pct:.0f}% ({i+1}/{user_number} users)")

        # ------------------------------------------------------------------
        # Phase 2: CF -- build neighbor data, then run CF for remaining time
        # ------------------------------------------------------------------
        if epr_fraction < 1.0:
            cf_updates   = 4  ##NEED MAIN IMPLEMENTATION!!!!
            n_intervals  = cf_updates + 1
            cf_duration  = time - epr_time
            interval_len = cf_duration / n_intervals

            check_run_dir = None  # created on first interval, reused afterwards

            for interval in range(n_intervals):
                t_start = epr_time + interval * interval_len
                t_end   = epr_time + (interval + 1) * interval_len

                logger.info(
                    f"CF interval {interval + 1}/{n_intervals}: "
                    f"rebuilding similarities, running [{t_start:.1f}, {t_end:.1f}]..."
                )
                cf_neighbors, check_run_dir = model.utils_user_knn.build_cf_data(
                    users_list     = self.users_list,
                    top_k          = top_k,
                    simulate_params= self._simulate_params,
                    grid_exists    = self.grid_exists,
                    grid_params    = getattr(self, "_grid_params", None),
                    check_run_dir  = check_run_dir,
                    interval_idx   = interval,
                )

                for i, user in enumerate(self.users_list):
                    neighbors = cf_neighbors[i]
                    while user.clock < t_end:
                        user.make_decision_cf(neighbors)

        for user in self.users_list:
            header    = user.user_history[0]
            clock_idx = header.index("Clock")
            state_idx = header.index("State(E/R)")
            if len(user.user_history) > 1 and user.user_history[-1][clock_idx] > time:
                last_row = user.user_history.pop()
                if last_row[state_idx] == "E":
                    user.venue_history.pop()

    def resume(self, simulate_params, grid_params=None):
        """
        Continue the simulation on users already in self.users_list.

        Users are advanced until their clock reaches simulate_params["time"],
        which is interpreted as an *absolute* target clock, not an additional
        duration.  This means the value should be larger than the clock values
        the users already carry (i.e. the end-time of the previous run phase).

        Supports the same EPR / EPR+CF logic as run():
          - epr_fraction == 1.0  →  pure EPR for the whole resume window
          - epr_fraction <  1.0  →  EPR up to (epr_fraction * extra_time), then CF

        where extra_time = simulate_params["time"] - mean clock of users at entry.

        This method updates self._simulate_params so that save_raw_data() and
        run_analysis() will use the new parameters (including the new "time")
        for folder naming.

        Parameters
        ----------
        simulate_params : dict
            Must contain: "ro", "gamma", "a_param", "b_param", "time".
            Optional: "epr_fraction" (float, default 1.0), "top_k" (int, default 10).
            "user_number" is ignored: all users already in memory are continued.
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
        a_param      = simulate_params["b_param"]   # kept for symmetry; not used in loop
        b_param      = simulate_params["b_param"]
        time_target  = simulate_params["time"]
        epr_fraction = simulate_params.get("epr_fraction", 1.0)
        top_k        = simulate_params.get("top_k", 10)

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

        ##PER ORA LO LASCIO COSì: Si introduce un artificio nello scorrimento del tempo,
        ##Da verificare se è problematico o no
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
            f"→ target={time_target}, epr_fraction={epr_fraction}"
        )

        if epr_fraction == 1.0:
            # ---------------------------------------------------------------
            # Pure EPR resume
            # ---------------------------------------------------------------
            for i, user in enumerate(self.users_list):
                while user.clock < time_target:
                    user.make_decision()
                if (i + 1) % milestone == 0 or (i + 1) == n_users:
                    pct = ((i + 1) / n_users) * 100
                    logger.info(f"Resume EPR progress: {pct:.0f}% ({i+1}/{n_users} users)")

        else:
            # ---------------------------------------------------------------
            # EPR phase up to epr_fraction * extra_time, then CF
            # ---------------------------------------------------------------
            epr_end = resume_start + epr_fraction * extra_time

            logger.info(f"Resume EPR phase: running to clock={epr_end:.1f}...")
            for i, user in enumerate(self.users_list):
                while user.clock < epr_end:
                    user.make_decision()
                if (i + 1) % milestone == 0 or (i + 1) == n_users:
                    pct = ((i + 1) / n_users) * 100
                    logger.info(f"Resume EPR progress: {pct:.0f}% ({i+1}/{n_users} users)")

            # CF intervals
            ##NEED MAIN IMPLEMENTATION!!!!
            cf_updates   = 4
            n_intervals  = cf_updates + 1
            cf_duration  = time_target - epr_end
            interval_len = cf_duration / n_intervals

            check_run_dir = None  # created on first interval, reused afterwards

            for interval in range(n_intervals):
                t_end = epr_end + (interval + 1) * interval_len
                logger.info(
                    f"Resume CF interval {interval + 1}/{n_intervals}: "
                    f"rebuilding similarities, running to {t_end:.1f}..."
                )
                cf_neighbors, check_run_dir = model.utils_user_knn.build_cf_data(
                    users_list     = self.users_list,
                    top_k          = top_k,
                    simulate_params= self._simulate_params,
                    grid_exists    = self.grid_exists,
                    grid_params    = getattr(self, "_grid_params", None),
                    check_run_dir  = check_run_dir,
                    interval_idx   = interval,
                )
                for i, user in enumerate(self.users_list):
                    neighbors = cf_neighbors[i]
                    while user.clock < t_end:
                        user.make_decision_cf(neighbors)

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
            "b_param", "time", "grid", "random_start" (optional).
            Additional key:
                "batch_id" (str, optional)
                    Exact timestamp folder name to load (e.g. "20240601_120000_123456").
                    When absent, the most-recent batch folder is selected automatically.
        grid_params : dict, optional
            Required when load_params["grid"] is True.
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

        grid_exists  = load_params.get("grid", False)
        random_start = load_params.get("random_start", False)
        batch_id     = load_params.get("batch_id", None)

        # ------------------------------------------------------------------
        # Locate the parameter folder in raw_data/
        # ------------------------------------------------------------------
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, "raw_data")

        efrac  = float(load_params.get("epr_fraction", 1.0))
        rstart = load_params.get("random_start", False)

        if grid_exists and grid_params is not None:
            param_folder_name = (
                f"ro_{load_params['ro']}_g_{load_params['gamma']}"
                f"_a_{load_params['a_param']}_b_{load_params['b_param']}"
                f"_t_{load_params['time']}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_rs_{'y' if rstart else 'n'}"
                f"_ef_{efrac}"
            )
        else:
            param_folder_name = (
                f"ro_{load_params['ro']}_g_{load_params['gamma']}"
                f"_a_{load_params['a_param']}_b_{load_params['b_param']}"
                f"_t_{load_params['time']}"
                f"_rs_{'y' if rstart else 'n'}"
                f"_ef_{efrac}"
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
        self.ro          = load_params["ro"]
        self.gamma       = load_params["gamma"]
        self.a_param     = load_params["a_param"]
        self.grid_exists = grid_exists
        self._random_start = random_start

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
        venue_header = ["Venue_name", "Number of visits", "Venue_x_coord", "Venue_y_coord"]

        for pos, i in enumerate(user_indices):
            h_path = os.path.join(run_folder, f"user_{i}_history.parquet")
            v_path = os.path.join(run_folder, f"user_{i}_venue.parquet")
            g_path = os.path.join(run_folder, f"user_{i}_grid.parquet")

            history_df = pd.read_parquet(h_path)
            venue_df   = pd.read_parquet(v_path) if os.path.exists(v_path) else None

            # Rebuild history as list-of-lists (matching the live format)
            user_history  = [history_header] + history_df.values.tolist()
            venue_history = [venue_header]
            if venue_df is not None:
                venue_history += venue_df.values.tolist()

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
        user_{i}_history.parquet      : full user trajectory (copy from raw)
        user_{i}_venue.parquet        : venue visit counts   (copy from raw)
        user_{i}_stats.parquet        : rg, rg_two, x_cm, y_cm
        user_{i}_rg_over_time.parquet : clock vs rg(t) series
        user_{i}_s_over_time.parquet  : clock vs S(t) series

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
                filters = {k: v for k, v in analyse_params.items() if k != "grid"}
                if analyse_params["grid"] == True:
                    filters.update(grid_params)
                loaded = model.utils.load_raw_parquet(
                    filters=filters,
                    grid = analyse_params["grid"],
                    return_run_folders=True,
                )
                self.users_list = [entry["data"] for entry in loaded]
                raw_folders_to_lock.update(entry["run_folder"] for entry in loaded)
                logger.info(f"Loaded {len(self.users_list)} users from raw_data")
        elif raw_run_folder is not None:
            # simulate_and_analyse: users in memory, raw folder still known
            raw_folders_to_lock.add(raw_run_folder)

        if combined:
            if not hasattr(self, "_load_params") or self._load_params is None:
                raise RuntimeError(
                    "run_analysis(combined=True) requires _load_params to be set. "
                    "Call _reconstruct_users_from_raw() before this method."
                )
            analysis_folder = self._make_combined_run_folder(
                "combined_analysis_data",
                base_params   = self._load_params,
                resume_params = analyse_params,
                grid_params   = grid_params,
            )
        elif analyse_params["grid"] == True:
            analysis_folder = self._make_run_folder("analysis_data", analyse_params, grid_params)
        else:
            analysis_folder = self._make_run_folder("analysis_data", analyse_params)
        total     = len(self.users_list)
        milestone = max(1, total // 10)

        #List of raw data
        user_history_df_list = []
        venue_history_df_list = []

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

            stats_df        = model.utils.compute_stats(user_history_df)
            rg_over_time_df = model.utils.compute_rg_over_time(user_history_df)
            s_over_time_df  = model.utils.compute_s_over_time(user_history_df)

            user_history_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_history.parquet"),      index=False)
            venue_history_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_venue.parquet"),        index=False)
            stats_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_stats.parquet"),        index=False)
            rg_over_time_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_rg_over_time.parquet"), index=False)
            s_over_time_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_s_over_time.parquet"),  index=False)

            if analyse_params.get("grid", False):
                if isinstance(user_entry, model.user.User) and user_entry.has_grid:
                    grid_df = pd.DataFrame(user_entry.grid.matrix)
                elif isinstance(user_entry, dict) and user_entry.get("grid") is not None:
                    grid_df = user_entry["grid"]
                else:
                    grid_df = None
                if grid_df is not None:
                    grid_df.columns = grid_df.columns.astype(str)
                    grid_df.to_parquet(
                        os.path.join(analysis_folder, f"user_{i}_grid.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Analysis progress: {pct:.0f}% ({i+1}/{total} users)")

        gini_ind  = model.utils.compute_individual_gini(venue_history_df_list)
        gini_coll = model.utils.compute_collective_gini(venue_history_df_list)

        batch_stats_df = pd.DataFrame([
            {"metric": "gini_collective",     "value": gini_coll},
            {"metric": "gini_individual",     "value": gini_ind},
            {"metric": "n_users",             "value": float(total)},
            #{"metric": "n_venues_collective", "value": float(len(collective_venue_counts))},
            #collective_venue_counts is now inside the compute_collective_gini function, and its length is not returned.
        ])

        if analyse_params.get("grid", False):
            batch_grid, _ = model.utils.compute_and_save_aggregate_grid(analysis_folder)
            moran_i_dict  = model.utils.compute_moran_i(batch_grid)
            moran_df      = pd.DataFrame([{"metric": k, "value": v} for k, v in moran_i_dict.items()])
            batch_stats_df = pd.concat([batch_stats_df, moran_df], ignore_index=True)

        batch_stats_df.to_parquet(
            os.path.join(analysis_folder, "batch_stats.parquet"), index=False
        )

        logger.info(f"Analysis data saved to: {analysis_folder}")

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
            "grid" (bool), "random_start" (bool), "epr_fraction" (float).
        grid_params : dict, optional
            When provided (and plot_params["grid"] is True), must contain
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
            "mobility":          {"user_history", "venue_history", "stats"},
            "rg_distribution":   {"stats"},
            "freq_distribution": {"venue_history"},
            "two_radius":        {"stats"},
            "s_over_time":       {"s_over_time"},
            "rg_over_time":      {"rg_over_time"},
            "aggregate_grid":    set(),   # only aggregate parquet, no per-user data
            "batch_grid":        set(),
            "batch_stats":       set(),   # only batch_stats.parquet, no per-user data
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

        filters = {**plot_params}
        if plot_params.get("grid") and grid_params:
            filters.update(grid_params)

        # gamma needed by freq_distribution plot
        self.gamma = plot_params.get("gamma")

        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []
        self.all_s_over_time = []

        # ------------------------------------------------------------------
        # Per-user data load (skipped when only grid plots are requested)
        # ------------------------------------------------------------------
        loaded = []
        if needs_user_data:
            loaded = model.utils.load_analysis_parquet(filters=filters, needed_files=needed_files)

        # Keep a copy of the full filters with the actual user count for titles/file names.
        # _plot_grid_params excludes user_number (irrelevant for grid/aggregate plots).
        self._plot_params = {**filters, "user_number": len(loaded)}
        self._plot_grid_params = {k: v for k, v in filters.items() if k != "user_number"}

        for entry in loaded:
            stats_df         = entry.get("stats")
            venue_history_df = entry.get("venue_history")
            rg_over_time_df  = entry.get("rg_over_time")
            s_over_time_df   = entry.get("s_over_time")
            user_history_df  = entry.get("user_history")

            self.last_user_history  = user_history_df
            self.last_venue_history = venue_history_df
            self.last_stats         = stats_df

            if stats_df is not None:
                self.rg_list.append(
                    stats_df.loc[stats_df['metric'] == "radius of gyration",     'value'].iloc[0])
                self.two_rg_list.append(
                    stats_df.loc[stats_df['metric'] == "radius of gyration two", 'value'].iloc[0])
                self.last_x_cm = stats_df.loc[stats_df['metric'] == "x_center of mass", 'value'].iloc[0]
                self.last_y_cm = stats_df.loc[stats_df['metric'] == "y_center of mass", 'value'].iloc[0]

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
                    mask_center= p["grid"] and not p.get("random_start", False),
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
                        mask_center= p["grid"] and not p.get("random_start", False),
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

    # ==========================================================================
    # UTILITY
    # ==========================================================================

    def _make_run_folder(self, data_type, params, grid_params=None):
        """
        Create and return a timestamped run folder under a data_type root.

        Folder hierarchy:
            <project_root>/<data_type>/
                ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                    {YYYYMMDD_HHMMSS_ffffff}/

        When grid_params is provided, the parameter folder is extended with
        grid dimensions:
                ro_{ro}_..._time_{time}_clen_{cell_lenght}_cnum_{cell_number}/

        The single-token keys ``clen``, ``cnum``, and ``rstart`` are used
        so that the folder-name parser in the data loaders can split on
        ``_`` and pair tokens unambiguously.

        When a grid is active and ``random_start=True`` is present in
        params, the suffix ``_rstart_yes`` is appended to distinguish
        random-origin runs from fixed-origin ones.

        When ``epr_fraction`` is present in params and < 1.0, the suffix
        ``_efrac_{epr_fraction}`` is appended to distinguish CF runs from
        pure EPR runs.

        Parameters
        ----------
        data_type : str
            e.g. "raw_data" or "analysis_data".
        params : dict
            Must contain "ro", "gamma", "a_param", "b_param", "time".
            Optional: "random_start" (bool), "epr_fraction" (float).
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, data_type)

        efrac  = float(params.get("epr_fraction", 1.0))
        rstart = params.get("random_start", False)

        if grid_params is None:
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_g_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_t_{params['time']}"
                f"_rs_{'y' if rstart else 'n'}"
                f"_ef_{efrac}"
            )
        else:
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_g_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_t_{params['time']}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_rs_{'y' if rstart else 'n'}"
                f"_ef_{efrac}"
            )
        run_folder = os.path.join(param_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def _make_combined_run_folder(self, data_type, base_params, resume_params, grid_params=None):
        """
        Create and return a timestamped run folder under a two-level parameter
        hierarchy for chained (load → resume) simulations.

        Folder hierarchy
        ----------------
        <project_root>/<data_type>/
            <base_param_folder>/        ← built from base_params  (the loaded batch)
                <resume_param_folder>/  ← built from resume_params (the new simulation)
                    {YYYYMMDD_HHMMSS_ffffff}/

        Both param folder names follow the same naming convention as
        _make_run_folder(): ro / gamma / a / b / time [/ clen / cnum]
        [/ rstart_yes] [/ efrac_X].

        Parameters
        ----------
        data_type : str
            Root directory name, e.g. "combined_raw_data" or
            "combined_analysis_data".
        base_params : dict
            Parameters of the original loaded batch.
            Must contain "ro", "gamma", "a_param", "b_param", "time".
        resume_params : dict
            Parameters used for the resume phase.
            Must contain "ro", "gamma", "a_param", "b_param", "time".
        grid_params : dict, optional
            When provided, must contain "cell_lenght" and "cell_number".
            Applied to both folder levels.

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_folder  = os.path.join(project_root, data_type)

        # Shared physical params (must match between base and resume)
        ro      = base_params["ro"]
        gamma   = base_params["gamma"]
        a_param = base_params["a_param"]
        b_param = base_params["b_param"]
        rstart  = base_params.get("random_start", False)

        t0 = base_params["time"]
        t1 = resume_params["time"]

        ef0 = float(base_params.get("epr_fraction", 1.0))
        ef1 = float(resume_params.get("epr_fraction", 1.0))
        ef_part = f"_ef0_{ef0}_ef1_{ef1}"

        if grid_params is not None:
            param_folder_name = (
                f"ro_{ro}_g_{gamma}"
                f"_a_{a_param}_b_{b_param}"
                f"_t0_{t0}_t1_{t1}"
                f"_cl_{grid_params['cell_lenght']}_cn_{grid_params['cell_number']}"
                f"_rs_{'y' if rstart else 'n'}"
                f"{ef_part}"
            )
        else:
            param_folder_name = (
                f"ro_{ro}_g_{gamma}"
                f"_a_{a_param}_b_{b_param}"
                f"_t0_{t0}_t1_{t1}"
                f"_rs_{'y' if rstart else 'n'}"
                f"{ef_part}"
            )

        run_folder = os.path.join(
            base_folder,
            param_folder_name,
            datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        )
        os.makedirs(run_folder, exist_ok=True)
        return run_folder