import numpy as np
import model.user
import model.utils
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
            # Determine starting position
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
            cf_updates   = 4 ##NEED MAIN IMPLEMENTATION!!!!
            n_intervals  = cf_updates + 1
            cf_duration  = time - epr_time
            interval_len = cf_duration / n_intervals

            for interval in range(n_intervals):
                t_start = epr_time + interval * interval_len
                t_end   = epr_time + (interval + 1) * interval_len

                logger.info(
                    f"CF interval {interval + 1}/{n_intervals}: "
                    f"rebuilding similarities, running [{t_start:.1f}, {t_end:.1f}]..."
                )
                cf_neighbors_list = self._build_cf_data(top_k)

                for i, user in enumerate(self.users_list):
                    neighbors = cf_neighbors_list[i]
                    while user.clock < t_end:
                        user.make_decision_cf(neighbors)

        # -----------------------------------------------------------------------
        # Trim: remove the last record if it pushed the clock past the time limit
        # -----------------------------------------------------------------------
        for user in self.users_list:
            header    = user.user_history[0]
            clock_idx = header.index("Clock")
            state_idx = header.index("State(E/R)")
            if len(user.user_history) > 1 and user.user_history[-1][clock_idx] > time:
                last_row = user.user_history.pop()
                if last_row[state_idx] == "E":
                    user.venue_history.pop()

    def _build_cf_data(self, top_k):
        """
        Build cosine-similarity-based top-k neighbor lists for every user.

        Constructs a user-venue visit-count matrix from the current state of
        self.users_list, computes pairwise cosine similarities, and returns for
        each user a ranked list of the most similar neighbors.

        Parameters
        ----------
        top_k : int
            Maximum number of neighbors to retain per user.
            If fewer users have positive similarity, all of them are kept.

        Returns
        -------
        list of list of (User, float)
            cf_neighbors[i] = [(neighbor_User, similarity_weight), ...]
            sorted by descending similarity for user i.
            An empty list is returned for users with no positively similar neighbor.

        Notes
        -----
        - Similarity is cosine similarity on raw visit counts.
        - Self-similarity is zeroed out (a user is never its own neighbor).
        - Only neighbors with strictly positive similarity are included.
        """
        n_users = len(self.users_list)

        # Collect all unique venue names in insertion order
        all_venues = []
        venue_set  = set()
        for user in self.users_list:
            for row in user.venue_history[1:]:
                if row[0] not in venue_set:
                    venue_set.add(row[0])
                    all_venues.append(row[0])
        venue_to_idx = {v: i for i, v in enumerate(all_venues)}
        n_venues = len(all_venues)

        # Build user-venue visit count matrix
        uv_matrix = np.zeros((n_users, n_venues), dtype=float)
        for ui, user in enumerate(self.users_list):
            for row in user.venue_history[1:]:
                uv_matrix[ui, venue_to_idx[row[0]]] = row[1]

        # Cosine similarity matrix
        norms = np.linalg.norm(uv_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        uv_norm    = uv_matrix / norms
        sim_matrix = uv_norm @ uv_norm.T
        np.fill_diagonal(sim_matrix, 0.0)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        check_dir = os.path.join(root_dir, "check")
        os.makedirs(check_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        np.savetxt(os.path.join(check_dir, f"sim_matrix_{ts}.txt"), sim_matrix, fmt='%.6f')

        top_k_actual = min(top_k, n_users - 1)
        cf_neighbors = []
        for ui in range(n_users):
            sims     = sim_matrix[ui]
            pos_mask = sims > 0
            if not pos_mask.any():
                cf_neighbors.append([])
                continue
            pos_indices = np.where(pos_mask)[0]
            if len(pos_indices) > top_k_actual:
                # Keep the top_k_actual most similar
                pos_indices = pos_indices[np.argsort(sims[pos_indices])[::-1]][:top_k_actual]
            else:
                pos_indices = pos_indices[np.argsort(sims[pos_indices])[::-1]]
            cf_neighbors.append(
                [(self.users_list[j], float(sims[j])) for j in pos_indices]
            )

        with open(os.path.join(check_dir, f"neighbors_{ts}.txt"), "w") as f:
            for row in cf_neighbors:
                f.write(",".join(map(str, row)) + "\n")

        logger.info(
            f"CF neighbor data built: {n_users} users, {n_venues} unique venues, "
            f"top_k={top_k_actual}"
        )
        return cf_neighbors

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
            #print(venue_df)

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

    # ==========================================================================
    # PHASE 2: ANALYSIS
    # ==========================================================================

    def run_analysis(self, analyse_params, raw_run_folder=None, grid_params=None):
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

        if analyse_params["grid"] == True:
            analysis_folder = self._make_run_folder("analysis_data", analyse_params, grid_params)
        else:
            analysis_folder = self._make_run_folder("analysis_data", analyse_params)
        total     = len(self.users_list)
        milestone = max(1, total // 10)

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

        if analyse_params.get("grid", False):
            model.utils.compute_and_save_aggregate_grid(analysis_folder)

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
        }

        if plot_list is not None:
            needed_files = set()
            for p in plot_list:
                needed_files |= _PLOT_FILE_NEEDS.get(p, set())
            needs_user_data = bool(needed_files)
            needs_grids     = any(p in ("aggregate_grid", "batch_grid") for p in plot_list)
        else:
            needed_files    = None   # load everything
            needs_user_data = True
            needs_grids     = True

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
        p = self._plot_params
        pg = self._plot_grid_params

        if "mobility" in plot_list:
            model.utils.plot_mobility_visualization(
                self.last_user_history, self.last_venue_history,
                stats=self.last_stats,
                filters=p, title=model.utils.build_plot_title("User Mobility", p),
                show=self._show
            )
        if "rg_distribution" in plot_list:
            model.utils.plot_rg_distribution(
                self.rg_list,
                filters=p, title=model.utils.build_plot_title("RG Distribution", p),
                show=self._show
            )
        if "freq_distribution" in plot_list:
            model.utils.plot_frequency_distribution(
                self.venue_history_list, self.gamma,
                filters=p, title=model.utils.build_plot_title("Frequency Distribution", p),
                show=self._show
            )
        if "two_radius" in plot_list:
            model.utils.plot_two_radius_vs_radius(
                self.rg_list, self.two_rg_list,
                filters=p, title=model.utils.build_plot_title("Two-RG vs RG", p),
                show=self._show
            )
        if "s_over_time" in plot_list:
            model.utils.plot_new_s_over_time_from_df(
                self.all_s_over_time, filters=p, bins_per_decade=10,
                title=model.utils.build_plot_title("S over time", p),
                show=self._show
            )
        if "rg_over_time" in plot_list:
            model.utils.plot_r_gyr_over_time_from_df(
                self.all_rg_over_time, filters=p, bins_per_decade=10,
                title=model.utils.build_plot_title("Radius of Gyration over time", p),
                show=self._show
            )
        if "aggregate_grid" in plot_list:
            if self.aggregate_grid is not None:
                pg_with_n = {**pg, "u_n": self.aggregate_grid_user_count}
                model.utils.plot_aggregate_grid(
                    self.aggregate_grid, filters=pg_with_n,
                    title=model.utils.build_plot_title("Aggregate Grid", pg_with_n),
                    mask_center= p["grid"] and not p.get("random_start", False),
                    show=self._show
                )
            else:
                logger.warning("No param-level aggregate grid available to plot.")
        if "batch_grid" in plot_list:
            if self.batch_grids:
                for batch_id, grid, batch_user_count in self.batch_grids:
                    pg_with_n = {**pg, "u_n": batch_user_count}
                    model.utils.plot_aggregate_grid(
                        grid, filters=pg_with_n,
                        title=model.utils.build_plot_title(f"Grid - {batch_id}", pg_with_n),
                        batch_id=batch_id,
                        mask_center= p["grid"] and not p.get("random_start", False),
                        show=self._show
                    )
            else:
                logger.warning("No batch-level aggregate grids available to plot.")

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

        efrac_suffix = (
            f"_efrac_{params['epr_fraction']}"
            if params.get("epr_fraction", 1.0) < 1.0
            else ""
        )

        if grid_params is None:
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_gamma_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_time_{params['time']}"
                f"{efrac_suffix}"
        )
            
        else:
            rstart_suffix = "_rstart_yes" if params.get("random_start", False) else ""
            param_folder = os.path.join(
                base_folder,
                f"ro_{params['ro']}_gamma_{params['gamma']}"
                f"_a_{params['a_param']}_b_{params['b_param']}_time_{params['time']}"
                f"_clen_{grid_params['cell_lenght']}_cnum_{grid_params['cell_number']}"
                f"{rstart_suffix}"
                f"{efrac_suffix}"
        )    
        run_folder = os.path.join(param_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_folder, exist_ok=True)
        return run_folder