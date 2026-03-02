import numpy as np
import model.user
import model.utils
import pandas as pd
import os
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


class Simulation:
    """
    Orchestrates the three-phase mobility simulation pipeline:

    Phase 1 - SIMULATE : run the EPR model for N users, save raw trajectories to raw_data/
    Phase 2 - ANALYSIS : load raw trajectories, compute analysis quantities, save to analysis_data/
    Phase 3 - PLOT     : load analysis data and generate all requested plots

    Phases can be run independently or chained.  When simulate_and_analyse() is called,
    Phase 2 operates directly on the in-memory User objects produced by Phase 1, avoiding
    a round-trip to disk.
    """

    def __init__(self, ro, gamma, a_param, b_param, user_number, time,
                 plots_to_generate=None, filters=None):
        """
        Parameters
        ----------
        ro : float
            Base exploration probability.
        gamma : float
            Exploration decay exponent.
        a_param : float
            Spatial power-law exponent.
        b_param : float
            Temporal power-law exponent.
        user_number : int
            Number of users to simulate.
        time : float
            Simulation time horizon per user.
        plots_to_generate : list of str, optional
            Names of plots to generate in the PLOT phase.
            Accepted values: "mobility", "rg_distribution", "freq_distribution",
            "two_radius", "s_over_time", "rg_over_time".
        filters : dict, optional
            Parameter filters used when loading saved data.
            Keys: "ro", "gamma", "a_param", "b_param", "time", "user_number".
            "user_number" is special: positive = first N, negative = last N, absent = all.
        """
        self.ro = ro
        self.gamma = gamma
        self.a_param = a_param
        self.b_param = b_param
        self.user_number = user_number
        self.time = time
        self.plots_to_generate = plots_to_generate or []
        self.filters = filters

        # Populated during run() — holds User objects
        self.users_list = []

        # Populated during load_analysis_data() — used by plot()
        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []   # list of DataFrames, one per user
        self.all_s_over_time = []    # list of DataFrames, one per user

        # Single-user references for mobility / freq_distribution plots
        self.last_user_history = None
        self.last_venue_history = None
        self.last_stats = None

    # ==========================================================================
    # PHASE 1: SIMULATE
    # ==========================================================================

    def run(self):
        """
        Run the EPR simulation for self.user_number users up to self.time each.
        Results are stored in self.users_list as User objects.
        """
        milestone = max(1, self.user_number // 10)
        logger.info(f"Starting simulation for {self.user_number} users...")

        for i in range(self.user_number):
            user = model.user.User(
                uid=f"User_{i}",
                a_param=self.a_param,
                b_param=self.b_param,
                simulation=self
            )
            while user.clock < self.time:
                user.make_decision()

            self.users_list.append(user)

            if (i + 1) % milestone == 0 or (i + 1) == self.user_number:
                pct = ((i + 1) / self.user_number) * 100
                logger.info(f"Simulation progress: {pct:.0f}% ({i+1}/{self.user_number} users)")

    def save_raw_data(self, base_folder=None):
        """
        Serialise user_history and venue_history for every simulated user to
        Parquet files under raw_data/.

        No statistical quantities are computed here.

        Folder layout
        -------------
        raw_data/
            ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                {YYYYMMDD_HHMMSS_ffffff}/
                    user_0_history.parquet
                    user_0_venue.parquet
                    ...

        Parameters
        ----------
        base_folder : str, optional
            Override for the raw_data root directory.

        Returns
        -------
        str
            Path to the timestamped run folder that was just created.
        """
        logger.info("Saving raw simulation data...")

        run_folder = self.make_run_folder("raw_data", base_folder)
        total = len(self.users_list)
        milestone = max(1, total // 10)

        for i, user in enumerate(self.users_list):
            user_df = pd.DataFrame(user.user_history[1:], columns=user.user_history[0])
            venue_df = pd.DataFrame(user.venue_history[1:], columns=user.venue_history[0])

            user_df.to_parquet(os.path.join(run_folder, f"user_{i}_history.parquet"), index=False)
            venue_df.to_parquet(os.path.join(run_folder, f"user_{i}_venue.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Save progress: {pct:.0f}% ({i+1}/{total} users saved)")

        logger.info(f"Raw data saved to: {run_folder}")
        return run_folder

    # ==========================================================================
    # PHASE 2: ANALYSIS
    # ==========================================================================

    def run_analysis(self, raw_run_folder=None, base_folder=None):
        """
        Compute all analysis quantities from raw trajectory data and save them
        to analysis_data/.

        This method handles two calling patterns:

        (a) Chained after run() + save_raw_data()
            Pass the folder path returned by save_raw_data() as raw_run_folder.
            If users_list is still populated from run(), no disk read is performed
            at all — the User objects are converted directly.

        (b) Standalone (ANALYSE mode)
            Leave raw_run_folder=None.  Data is loaded from raw_data/ using
            self.filters.

        Files saved per user
        --------------------
        user_{i}_history.parquet   : full user trajectory (copy from raw)
        user_{i}_venue.parquet     : venue visit counts   (copy from raw)
        user_{i}_stats.parquet     : rg, rg_two, x_cm, y_cm
        user_{i}_rg_over_time.parquet : clock vs rg(t) series
        user_{i}_s_over_time.parquet  : clock vs S(t) series

        Parameters
        ----------
        raw_run_folder : str, optional
            Path to a specific raw_data run folder.  Ignored when users_list
            is already populated.
        base_folder : str, optional
            Override for the analysis_data root directory.

        Returns
        -------
        str
            Path to the timestamped analysis run folder that was just created.
        """
        logger.info("Starting analysis phase...")

        # Load from disk only when there are no in-memory users
        if not self.users_list:
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
                logger.info(f"Loaded {i} users from raw run folder")
            else:
                logger.info("Loading raw data via filters...")
                loaded = model.utils.load_raw_parquet(filters=self.filters, base_folder=base_folder)
                self.users_list = list(loaded)   # list of (user_history_df, venue_history_df)
                logger.info(f"Loaded {len(self.users_list)} users from raw_data")

        analysis_folder = self.make_run_folder("analysis_data", base_folder)
        total = len(self.users_list)
        milestone = max(1, total // 10)

        for i, user_entry in enumerate(self.users_list):
            # Support both live User objects and (df, df) tuples loaded from disk
            if isinstance(user_entry, model.user.User):
                user_history_df = pd.DataFrame(
                    user_entry.user_history[1:], columns=user_entry.user_history[0]
                )
                venue_history_df = pd.DataFrame(
                    user_entry.venue_history[1:], columns=user_entry.venue_history[0]
                )
            else:
                user_history_df, venue_history_df = user_entry

            stats_df        = compute_stats(user_history_df)
            rg_over_time_df = compute_rg_over_time(user_history_df)
            s_over_time_df  = compute_s_over_time(user_history_df)

            user_history_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_history.parquet"), index=False)
            venue_history_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_venue.parquet"), index=False)
            stats_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_stats.parquet"), index=False)
            rg_over_time_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_rg_over_time.parquet"), index=False)
            s_over_time_df.to_parquet(
                os.path.join(analysis_folder, f"user_{i}_s_over_time.parquet"), index=False)

            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = ((i + 1) / total) * 100
                logger.info(f"Analysis progress: {pct:.0f}% ({i+1}/{total} users)")

        logger.info(f"Analysis data saved to: {analysis_folder}")
        return analysis_folder

    def load_analysis_data(self, base_folder=None):
        """
        Load pre-computed analysis data from analysis_data/ into memory, ready
        for the PLOT phase.

        Uses self.filters to locate the matching parameter folder.  After loading,
        self.rg_list, self.two_rg_list, self.venue_history_list, self.all_rg_over_time
        and self.all_s_over_time are populated, and self.last_* attributes hold
        the data for the last loaded user (used by single-user plots).

        Parameters
        ----------
        base_folder : str, optional
            Override for the analysis_data root directory.
        """
        logger.info("Loading analysis data...")

        loaded = model.utils.load_analysis_parquet(filters=self.filters, base_folder=base_folder)

        if self.filters is None:
            self.filters = {}
        self.filters['user_number'] = len(loaded)
        logger.info(f"Loaded {len(loaded)} users from analysis_data")

        self.rg_list = []
        self.two_rg_list = []
        self.venue_history_list = []
        self.all_rg_over_time = []
        self.all_s_over_time = []

        for entry in loaded:
            stats_df        = entry.get("stats")
            venue_history_df = entry.get("venue_history")
            rg_over_time_df = entry.get("rg_over_time")
            s_over_time_df  = entry.get("s_over_time")
            user_history_df = entry.get("user_history")

            # Update single-user references (last loaded wins)
            self.last_user_history  = user_history_df
            self.last_venue_history = venue_history_df
            self.last_stats         = stats_df

            if stats_df is not None:
                self.rg_list.append(
                    stats_df.loc[stats_df['metric'] == "radius of gyration", 'value'].iloc[0])
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

    # ==========================================================================
    # PHASE 3: PLOT
    # ==========================================================================

    def plot(self):
        """
        Generate all plots requested in self.plots_to_generate.

        Must be called after load_analysis_data() (PLOT mode) or after
        run_analysis() if plotting immediately after analysis.
        """
        if "mobility" in self.plots_to_generate:
            model.utils.plot_mobility_visualization(
                self.last_user_history,
                self.last_venue_history,
                stats=self.last_stats,
                filters=self.filters,
                title=model.utils.build_plot_title("User Mobility", self.filters)
            )

        if "rg_distribution" in self.plots_to_generate:
            model.utils.plot_rg_distribution(
                self.rg_list,
                filters=self.filters,
                title=model.utils.build_plot_title("RG Distribution", self.filters)
            )

        if "freq_distribution" in self.plots_to_generate:
            model.utils.plot_frequency_distribution(
                self.last_venue_history,
                self.gamma,
                filters=self.filters,
                title=model.utils.build_plot_title("Frequency Distribution", self.filters)
            )

        if "two_radius" in self.plots_to_generate:
            model.utils.plot_two_radius_vs_radius(
                self.rg_list,
                self.two_rg_list,
                filters=self.filters,
                title=model.utils.build_plot_title("Two-RG vs RG", self.filters)
            )

        if "s_over_time" in self.plots_to_generate:
            model.utils.plot_S_over_time_from_df(
                self.all_s_over_time,
                filters=self.filters,
                bins_per_decade=10,
                title=model.utils.build_plot_title("S over time", self.filters)
            )

        if "rg_over_time" in self.plots_to_generate:
            model.utils.plot_r_gyr_over_time_from_df(
                self.all_rg_over_time,
                filters=self.filters,
                bins_per_decade=10,
                title=model.utils.build_plot_title("Radius of Gyration over time", self.filters)
            )

    # ==========================================================================
    # UTILITY
    # ==========================================================================

    def make_run_folder(self, data_type, base_folder=None):
        """
        Create and return a timestamped run folder under a data_type root.

        The folder hierarchy is:
            <base_folder>/
                ro_{ro}_gamma_{gamma}_a_{a_param}_b_{b_param}_time_{time}/
                    {YYYYMMDD_HHMMSS_ffffff}/

        Parameters
        ----------
        data_type : str
            Subdirectory name under the project root, e.g. "raw_data" or "analysis_data".
            Used only when base_folder is None.
        base_folder : str, optional
            Explicit root path, overrides the default project-root/<data_type> location.

        Returns
        -------
        str
            Path to the newly created timestamped run folder.
        """
        if base_folder is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_folder = os.path.join(project_root, data_type)

        param_folder = os.path.join(
            base_folder,
            f"ro_{self.ro}_gamma_{self.gamma}_a_{self.a_param}_b_{self.b_param}_time_{self.time}"
        )
        run_folder = os.path.join(param_folder, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_folder, exist_ok=True)

        return run_folder


# =============================================================================
# ANALYSIS COMPUTATION FUNCTIONS
# These are module-level (not methods) because they depend only on DataFrames,
# not on Simulation state.  They can therefore be unit-tested or called
# independently of any Simulation instance.
# =============================================================================

def compute_stats(user_history_df):
    """
    Compute trajectory statistics from a user_history DataFrame.

    Computes:
      - x/y centre of mass (full trajectory)
      - x/y centre of mass (top-2 most visited venues only)
      - radius of gyration (full)
      - radius of gyration (top-2 venues)

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Destination', 'Destination_x_coord',
        'Destination_y_coord'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['metric', 'value'].
    """
    x_traj = np.concatenate(([0], user_history_df["Destination_x_coord"].values))
    y_traj = np.concatenate(([0], user_history_df["Destination_y_coord"].values))

    x_cm = np.mean(x_traj)
    y_cm = np.mean(y_traj)
    r_gyr = np.sqrt(np.mean((x_traj - x_cm)**2 + (y_traj - y_cm)**2))

    # Top-2 most visited destinations
    top2 = user_history_df["Destination"].value_counts().head(2).index.tolist()
    mask = user_history_df["Destination"].isin(top2)
    x_two = np.concatenate(([0], user_history_df.loc[mask, "Destination_x_coord"].values))
    y_two = np.concatenate(([0], user_history_df.loc[mask, "Destination_y_coord"].values))

    if len(x_two) > 1:
        x_cm_two = np.mean(x_two)
        y_cm_two = np.mean(y_two)
        r_gyr_two = np.sqrt(np.mean((x_two - x_cm_two)**2 + (y_two - y_cm_two)**2))
    else:
        x_cm_two, y_cm_two, r_gyr_two = x_cm, y_cm, 0.0

    return pd.DataFrame([
        {"metric": "x_center of mass",      "value": x_cm},
        {"metric": "y_center of mass",       "value": y_cm},
        {"metric": "x_center of mass two",   "value": x_cm_two},
        {"metric": "y_center of mass two",   "value": y_cm_two},
        {"metric": "radius of gyration",     "value": r_gyr},
        {"metric": "radius of gyration two", "value": r_gyr_two},
    ])


def compute_rg_over_time(user_history_df):
    """
    Compute the radius of gyration at every time step using an incremental
    (online) algorithm — O(n) instead of the naive O(n²).

    The incremental update uses the identity:
        Var(x_{1..n}) = E[x²] - E[x]²
    maintained via running sums of x, y, x², y² which are updated in O(1)
    per step.

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Clock', 'Destination_x_coord',
        'Destination_y_coord'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['clock', 'rg'].
    """
    clocks = user_history_df["Clock"].values
    x_vals = user_history_df["Destination_x_coord"].values
    y_vals = user_history_df["Destination_y_coord"].values

    n = len(x_vals)
    rg_values = np.empty(n)

    # Initialise with the fixed starting point (0, 0)
    sum_x  = 0.0
    sum_y  = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    count  = 1          # the origin (0,0) is always part of the trajectory

    for i in range(n):
        sum_x  += x_vals[i]
        sum_y  += y_vals[i]
        sum_x2 += x_vals[i] ** 2
        sum_y2 += y_vals[i] ** 2
        count  += 1

        mean_x = sum_x / count
        mean_y = sum_y / count

        # Var = E[x²] - E[x]²  (includes origin via initial sums = 0)
        var_x = sum_x2 / count - mean_x ** 2
        var_y = sum_y2 / count - mean_y ** 2

        rg_values[i] = np.sqrt(var_x + var_y)

    return pd.DataFrame({"clock": clocks, "rg": rg_values})


def compute_s_over_time(user_history_df):
    """
    Compute S(t) — the cumulative number of distinct venues visited — at each
    exploration event.

    S starts at 2 because the user begins at Venue_1 and the first action is
    always an exploration to Venue_2.

    Parameters
    ----------
    user_history_df : pd.DataFrame
        User history with columns 'Clock' and 'State(E/R)'.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ['clock', 's'].
    """
    e_rows = user_history_df[user_history_df["State(E/R)"] == "E"]
    clocks   = e_rows["Clock"].values
    s_values = np.arange(2, 2 + len(clocks))
    return pd.DataFrame({"clock": clocks, "s": s_values})