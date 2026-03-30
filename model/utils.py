import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import inspect
import scipy.stats

import logging
logger = logging.getLogger(__name__)

def power_law_sampling(par, min_val, max_val, n_samples):
    """
    Core function used to calculate samplings from analytical power law distribution p(x) ∝ x^(-1-par),
    as done in the EPR paper "Modelling the scaling properties of human mobility".
    Samples are generated following the Inverse transform sampling theorem. 
    
    Parameters:
    ----------
    par : float
        Power law exponent in p(x) ∝ x^(-1-par)
    min_val : float
        Minimum value. Has to be >0
    max_val : float
        Maximum value.
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled values from
        power law distribution
    """

    #logger.info(
    #    f"Generating {n_samples} samples from power-law "
    #    f"distribution (par={par}, min_val={min_val}, max_val={max_val})"
    #)


    u = np.random.uniform(0, 1, n_samples)

    #Using r values obtained from inversion of the cumulative distribution funtion
    if par != 1:
        r = (u * (min_val**(-par) - max_val**(-par)) + max_val**(-par))**(-1/par)

    #if alpha = 1 we have no more a power law
    else:
        r = u
    
    return r


def spatial_power_law_sampling(par = 0.55, min_val = 1, max_val = 100000, n_samples = 1):    
    """
    Wrapper funtion for power_law_sampling from a spatial distribution

    Parameters:
    ----------
    par : float, optional (default = 0.55)
        Power law exponent (0.55 in the paper for p(r) ∝ r^(-1-par))
    min_val : float, optional (default=5)
        Minimum value. Has to be >0. Chosen 5m arbitrarily as a unit-base for movement
    max_val : float, optional (default=100000)
        Maximum distance. Chosen 100km as in the paper for distance
    n_samples : int, optional (default=1)
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled distances from 
        the power law distribution p(r) ∝ r^(-1-par) 
    """
    #logger.info("Spatial sampling...")
    return power_law_sampling(par = par, min_val = min_val, max_val = max_val, n_samples = n_samples)

def temporal_power_law_sampling( par = 0.8, min_val = 1, max_val = 61200, n_samples = 1):    
    """
    Wrapper funtion for power_law_sampling from a temporal distribution 

    Parameters:
    ----------
    par : float, optional (default = 0.8)
        Power law exponent (0.8 in the paper for p(t) ∝ t^(-1-par))
    min_val : float, optional (default=300)
        Minimum value. Has to be >0. Chosen 300s (5 minutes) arbitrarily as a unit-base for waiting-time
    max_val : float, optional (default=100000)
        Maximum waiting-time. Chosen 61200s (17h) as in the paper
    n_samples : int, optional (default=1)
        Number of samples to generate

    Returns:
    --------
    samples : numpy.ndarray
        Array of shape (n_samples,) containing the sampled waiting-times from 
        the power law distribution p(t) ∝ t^(-1-par)
    """
    #logger.info("Temporal sampling...")
    return power_law_sampling(par = par, min_val = min_val, max_val = max_val, n_samples = n_samples)


def filepath_save_plot_or_show(folder = "test_plots", name: str = None, filters: dict = None, show = False):
    """
    Save function for plots (pdf format), which can be saved in different folders of the project.
    The name of the plot, if not passed, is automatically assigned as the name of the calling 
    function through the package 'inspect'

    Parameters:
    ----------
    folder : string, optional (default = "test_plots")
        name of the target folder. The folder must be in AI-E_PR folder.
    name : string, optional (default = None)
        name of the plot figure. If not passed the name of the calling function
        (the function which is calling 'filepath_save_plot', will be the name of the plot)
    filters : dict, optional (default = None)
        Dictionary of filters to be included in the plot name.

    Returns:
    --------
    None
        This function only produces plots for visual inspection and does not
        return any value.
    """
    if show:
        plt.show()
    else:

        # if name is not given, the calling function name is used instead
        if name is None:
            name = inspect.stack()[1].function

        # base folder 
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # folder where plots will be saved
        plots_dir = os.path.join(base_dir, f"{folder}")
        os.makedirs(plots_dir, exist_ok=True)

        filters = ", ".join(f"{k}={v}" for k, v in filters.items())
        filename = f"{name}_{filters}.pdf"
        filepath = os.path.join(plots_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

def build_plot_title(base_title, filters):
    """
    Build a plot title by combining a base title with simulation parameters.

    This function constructs a descriptive title for plots by appending loading
    parameters in a formatted string, using abbreviated parameter names.

    Parameters
    ----------
    base_title : str
        The base title for the plot.
    filters : dict
        Dictionary of filters to include in the title.

    Returns
    -------
    str
        The constructed title string.
    """
    _PARAM_ABBREV = {
        "ro":           "ρ",
        "gamma":        "γ",
        "a_param":      "a",
        "b_param":      "b",
        "time":         "t",
        "user_number":  "u_n",
        "cell_lenght":  "c_len",
        "cell_number":  "c_num",
        "grid":         "grid",
        "random_start": "r_s",
    }
    if not filters:
        return base_title

    param_str = ", ".join(f"{_PARAM_ABBREV.get(k, k)}={v}" for k, v in filters.items())
    return f"{base_title} ({param_str})"
        

def polar_to_cartesian (delta_r, angle, position=(0,0)):
    """
    Convert polar coordinates to Cartesian coordinates.

    This function transforms a distance and angle in polar coordinates
    to x and y coordinates in Cartesian space starting from a initial position.

    Parameters
    ----------
    delta_r : float
        The radial distance from the origin.
    angle : float
        The angle in radians from the positive x-axis.
    position : tuple of float, optional (default=(0,0))
        The initial position in Cartesian coordinates.

    Returns
    -------
    tuple of float
        A tuple (x, y) representing the Cartesian coordinates.
    """
    x = delta_r*np.cos(angle)
    y = delta_r*np.sin(angle)
    return (x + position[0], y + position[1])

def get_trajectory_arrays(user, k_top=None):
    """
    Extract trajectory coordinates from user history as NumPy arrays.

    This function processes the user's history data to extract x and y coordinates
    of the trajectory, optionally filtering to include only the top-k most visited venues.

    Parameters
    ----------
    user : User
        An object with user_history and venue_history as lists of lists.
    k_top : int, optional (default=None)
        Number of top visited venues to include. If None, include all venues.

    Returns
    -------
    x_traj : np.ndarray
        X coordinates of the trajectory, starting with 0.
    y_traj : np.ndarray
        Y coordinates of the trajectory, starting with 0.
    """
    header = user.user_history[0]
    data = user.user_history[1:]
    idx_dest_x = header.index("Destination_x_coord")
    idx_dest_y = header.index("Destination_y_coord")
    idx_dest_name = header.index("Destination")

    dest_x = np.array([row[idx_dest_x] for row in data], dtype=float)
    dest_y = np.array([row[idx_dest_y] for row in data], dtype=float)
    trajectory = [row[idx_dest_name] for row in data]

    if k_top is not None:
        venue_header = user.venue_history[0]
        venue_data = user.venue_history[1:]
        idx_venue_name = venue_header.index("Venue_name")
        idx_visits = venue_header.index("Number of visits")

        venue_visits = {row[idx_venue_name]: row[idx_visits] for row in venue_data}
        top_k_venues = sorted(venue_visits, key=venue_visits.get, reverse=True)[:k_top]
    else:
        top_k_venues = None 

    x_traj = [0]  
    y_traj = [0]

    for i, venue in enumerate(trajectory):
        if (top_k_venues is None) or (venue in top_k_venues):
            x_traj.append(dest_x[i])
            y_traj.append(dest_y[i])

    return np.array(x_traj), np.array(y_traj)


def dotted_histogram(numpy_array, bins_per_decade=4):
    """
    Compute a logarithmic histogram with density normalization.

    This function creates logarithmically spaced bins for the input array,
    computes frequencies, and normalizes to density. It filters out zero values
    and returns only bins with positive density.

    Parameters
    ----------
    numpy_array : array-like
        Input array of values to histogram (must be positive).
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic spacing.

    Returns
    -------
    centers_filtered : np.ndarray
        Geometric centers of the bins with positive density.
    density_filtered : np.ndarray
        Normalized density values for the bins.
    """
    array = numpy_array[numpy_array > 0]   # avoiding zeros

    min_array = array.min()
    max_array = array.max()

    log_min = np.log10(min_array)
    log_max = np.log10(max_array)

    #Avoiding boundary problems using a margin
    margin = 0.05
    log_min -= margin
    log_max += margin

    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))

    # Creation of logarithmic bins
    bins = np.logspace(log_min, log_max, num_bins + 1)

    # frequencies
    counts, edges = np.histogram(array, bins=bins)

    # bin logarithmic center
    centers = np.sqrt(edges[:-1] * edges[1:])

    # normailization
    widths = np.diff(edges)
    density = counts / widths / len(array)

    mask = density > 0 #Avoiding zero density bins 
    centers_filtered = centers[mask]
    density_filtered = density[mask]

    return centers_filtered, density_filtered


# =============================================================================
# ANALYSIS COMPUTATION FUNCTIONS
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
    x_traj = np.concatenate((user_history_df["Departure_x_coord"].values[0:1], user_history_df["Destination_x_coord"].values))
    y_traj = np.concatenate((user_history_df["Departure_y_coord"].values[0:1], user_history_df["Destination_y_coord"].values))

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
    Compute the radius of gyration at every time step

    The incremental update uses the identity:
        Var(x_{1..n}) = E[x²] - E[x]²
    maintained via running sums of x, y, x², y² 

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

    # Initialise with the starting point
    sum_x  = user_history_df["Departure_x_coord"].values[0]  
    sum_y  = user_history_df["Departure_y_coord"].values[0]  
    sum_x2 = user_history_df["Departure_x_coord"].values[0] ** 2
    sum_y2 = user_history_df["Departure_y_coord"].values[0] ** 2
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


def compute_gini_coefficient(visit_counts):
    """
    Compute the Gini coefficient for a distribution of visit counts,
    following Equation 1 in Pedreschi et al.:

        G = (1/|V|) * (|V| + 1 - 2 * (∑_{i=1}^{|V|} (|V|+1-i) * x_i) / ∑ x_i)

    where the x_i are sorted in ascending order and only venues with at
    least one visit are considered.

    A value of 0 means perfect equality (all venues visited equally);
    higher values indicate increasing concentration of visits on a smaller
    subset of venues.

    Parameters
    ----------
    visit_counts : array-like of int or float
        Raw visit counts for each venue.  Non-positive entries are ignored.

    Returns
    -------
    float
        Gini coefficient in [0, 1].  Returns 0.0 when fewer than two
        venues have been visited (degenerate case).
    """
    x = np.array(visit_counts, dtype=float)
    x = x[x > 0]           # keep only venues with ≥1 visit
    n = len(x)
    if n <= 1:
        return 0.0          # trivially equal with 0 or 1 venue
    x = np.sort(x)          # ascending order required by Eq. 1
    total = x.sum()
    if total == 0:
        return 0.0
    # ∑_{i=1}^{n} (n+1-i) * x_i   (i is 1-indexed)
    ranks = np.arange(1, n + 1)
    weighted_sum = np.dot((n + 1 - ranks), x)
    return (1.0 / n) * (n + 1 - 2.0 * weighted_sum / total)


def compute_individual_gini(venue_history_df):
    """
    Compute the individual Gini coefficient for a single user.

    Applies :func:`compute_gini_coefficient` to the per-venue visit counts
    recorded in the user's venue history (Equation 1 in Pedreschi et al.,
    restricted to the visits made by this user).

    Parameters
    ----------
    venue_history_df : pd.DataFrame
        User venue history with at least a 'Number of visits' column.

    Returns
    -------
    float
        Individual Gini coefficient in [0, 1].
    """
    visit_counts = venue_history_df["Number of visits"].values
    return compute_gini_coefficient(visit_counts)


def compute_collective_gini(collective_venue_counts):
    """
    Compute the collective Gini coefficient over the entire dataset.

    Applies :func:`compute_gini_coefficient` to the aggregate visit counts
    per venue, where each venue's count is the total number of times it was
    visited across *all* users in the dataset (Equation 1 in Pedreschi et al.).

    Parameters
    ----------
    collective_venue_counts : dict
        Mapping {venue_name: total_visit_count} aggregated across all users.

    Returns
    -------
    float
        Collective Gini coefficient in [0, 1].
    """
    visit_counts = list(collective_venue_counts.values())
    return compute_gini_coefficient(visit_counts)


# =============================================================================
# LOADERS
# =============================================================================

def load_parquet_folder(base_folder, filters, file_map, grid=False, return_run_folders=False, random_start=False, user_number=None):
    """
    Generic loader for per-user Parquet datasets stored under a parameter folder.

    This function is the single shared loading implementation used by both
    load_raw_parquet() and load_analysis_parquet(). It scans base_folder for
    parameter sub-folders that match the supplied filters, then loads the files
    described by file_map for each user found.

    Folder layout expected
    ----------------------
    base_folder/
        ro_{ro}_gamma_{gamma}_a_{a}_b_{b}_time_{time}/
            {run_id}/
                user_0_{suffix_1}.parquet
                user_0_{suffix_2}.parquet
                ...

    For grid simulations, the parameter folder is extended with grid dimensions:
        ro_{ro}_gamma_{gamma}_a_{a}_b_{b}_time_{time}_clen_{clen}_cnum_{cnum}/

    Parameters
    ----------
    base_folder : str
        Root directory to scan (e.g. the full path to raw_data/ or analysis_data/).
    filters : dict
        Parameter filters with None values, "user_number" and "grid" already removed.
        Keys should match simulation parameter names
        ("ro", "gamma", "a_param", "b_param", "time").
        For grid simulations, "cell_lenght" and "cell_number" may also be present.
    file_map : dict
        Maps a result-dict key to the per-user filename suffix.
        Example: {"user_history": "_history.parquet", "venue_history": "_venue.parquet"}
        The first key in file_map is used to discover which user indices exist in a run
        folder (i.e. it must be present for every user).
    grid : bool, optional (default=False)
        Whether to match grid simulation folders (containing clen/cnum tokens) or
        non-grid folders. Folders whose grid status does not match this flag are
        silently skipped, preventing cross-contamination between grid and non-grid runs
        with otherwise identical parameters.
    return_run_folders : bool, optional (default=False)
        When False, returns plain user-data dicts (keys from file_map).
        When True, wraps each entry as {"data": <user_dict>, "run_folder": <path>}
        so callers can trace which run folder each user came from.
    user_number : int or None, optional (default=None)
        When positive, loading stops as soon as this many users have been
        accumulated (early-exit — no unnecessary disk reads).
        When negative or None, all matching users are loaded; the final
        slicing for negative values is done by the caller via
        _apply_user_number_slice().

    Returns
    -------
    list of dict
        When return_run_folders is False: one dict per loaded user with keys
        matching file_map; values are DataFrames (or None when a file is absent).
        When return_run_folders is True: one dict per user with keys "data"
        (the user dict) and "run_folder" (str path to the source run folder).

    Notes
    -----
    Run folders that contain a ``_analyzed.lock`` file are silently skipped.
    Run folders are iterated in sorted order so that positive user_number
    always selects the same deterministic subset of users.
    The "grid" key is handled separately via the grid parameter and must not
    appear in filters; callers are responsible for removing it beforehand.
    """
    if grid == True:
        key_map = {
            "ro":           "ro",
            "gamma":        "gamma",
            "a_param":      "a",
            "b_param":      "b",
            "time":         "time",
            "cell_lenght":  "clen",
            "cell_number":  "cnum",
            "epr_fraction": "efrac",
        }
    elif grid == False:
        key_map = {
            "ro":           "ro",
            "gamma":        "gamma",
            "a_param":      "a",
            "b_param":      "b",
            "time":         "time",
            "epr_fraction": "efrac",
        }

    primary_suffix = next(iter(file_map.values()))

    matching_runs = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split("_")
        if len(parts) % 2 != 0:
            continue
        try:
            raw_params = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}
        except Exception:
            continue

        converted = {}
        for k, v in raw_params.items():
            if k == "time":
                try:
                    converted[k] = int(v)
                except ValueError:
                    converted[k] = v
            else:
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v

        # Scarta cartelle che non corrispondono alla modalità grid richiesta
        folder_has_grid = "clen" in converted and "cnum" in converted
        if grid != folder_has_grid:
            continue

        # Scarta cartelle che non corrispondono alla modalità random_start richiesta
        if grid:
            folder_has_rstart = converted.get("rstart") == "yes"
            if folder_has_rstart != bool(random_start):
                continue

        if any(converted.get(key_map.get(fk, fk)) != fv
               for fk, fv in filters.items()):
            continue

        for run_folder in sorted(os.listdir(folder_path)):  # sorted for determinism
            run_path = os.path.join(folder_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            if os.path.exists(os.path.join(run_path, "_analyzed.lock")):
                logger.debug(f"Skipping already-analyzed run folder: {run_path}")
                continue

            user_indices = []
            for fname in os.listdir(run_path):
                if fname.endswith(primary_suffix):
                    token = fname[len("user_") : -len(primary_suffix)]
                    try:
                        user_indices.append((int(token), token))
                    except ValueError:
                        user_indices.append((0, token))

            if user_indices:
                user_indices.sort(key=lambda x: x[0])
                matching_runs.append((run_path, [s for _, s in user_indices]))

    # Early-exit limit: for positive user_number stop as soon as N users are
    # accumulated, avoiding unnecessary disk reads.  For negative user_number
    # (last N) or None (all) we must load everything first.
    early_stop = user_number if (user_number is not None and user_number > 0) else None

    loaded = []
    done = False
    for run_path, user_indices in matching_runs:
        if done:
            break
        total_in_run = len(user_indices)
        milestone    = max(1, total_in_run // 10)
        for pos, idx in enumerate(user_indices):
            if early_stop is not None and len(loaded) >= early_stop:
                done = True
                break
            entry = {}
            for key, suffix in file_map.items():
                fpath = os.path.join(run_path, f"user_{idx}{suffix}")
                entry[key] = pd.read_parquet(fpath) if os.path.exists(fpath) else None
            if entry[next(iter(file_map))] is not None:
                if return_run_folders:
                    loaded.append({"data": entry, "run_folder": run_path})
                else:
                    loaded.append(entry)
            if (pos + 1) % milestone == 0 or (pos + 1) == total_in_run:
                pct = ((pos + 1) / total_in_run) * 100
                logger.info(
                    f"Loading progress: {pct:.0f}% ({pos+1}/{total_in_run} users) "
                    f"[{os.path.basename(run_path)}]"
                )

    return loaded

def load_raw_parquet(filters=None, grid = False, return_run_folders=False):
    """
    Load raw simulation data (user_history, venue_history) from raw_data/.

    Parameters
    ----------
    filters : dict, optional
        Parameter filters. Accepted keys: "ro", "gamma", "a_param", "b_param",
        "time", "user_number", "cell_lenght", "cell_number".
        "user_number": positive = first N users, negative = last N, absent = all.
        Include "cell_lenght" and "cell_number" to match grid simulation folders.
    return_run_folders : bool, optional (default=False)
        When False (default), returns a plain list of dicts with keys
        "user_history" and "venue_history" — backward-compatible behaviour.
        When True, returns a list of dicts with keys "data" (the user dict)
        and "run_folder" (the raw run folder path), so callers can lock
        folders after analysis.

    Returns
    -------
    list of dict
        See return_run_folders for the two possible shapes.
    """
    logger.info("Loading raw simulation data (raw_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)
    filters_clean = {k: v for k, v in filters.items()
                 if v is not None and k not in ("user_number", "grid", "random_start")
                 and not (k == "epr_fraction" and v == 1)}
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "raw_data")

    if not os.path.exists(base_folder):
        logger.warning(f"raw_data folder not found: {base_folder}")
        return []

    file_map = {
        "user_history":  "_history.parquet",
        "venue_history": "_venue.parquet",
    }

    grid = filters.get("grid", grid)
    if grid:
        file_map["grid"] = "_grid.parquet"

    loaded = load_parquet_folder(base_folder, filters_clean, file_map, grid=grid,
                                 return_run_folders=return_run_folders,
                                 random_start=filters.get("random_start", False),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Loaded {len(loaded)} users from raw_data")
    return loaded


def load_analysis_parquet(filters=None, needed_files=None):
    """
    Load analysis data from analysis_data/.

    Parameters
    ----------
    filters : dict, optional
        Parameter filters. Accepted keys: "ro", "gamma", "a_param", "b_param",
        "time", "user_number".
    needed_files : set of str, optional
        Subset of file-map keys to load (e.g. {"stats", "venue_history"}).
        When None (default), all files are loaded.

    Returns
    -------
    list of dict
        One dict per user with keys from the effective file_map.
        Any value may be None if the corresponding file is absent.
    """
    logger.info("Loading analysis data (analysis_data)...")

    if filters is None:
        filters = {}

    user_number   = filters.get("user_number", None)
    grid          = filters.get("grid", False)
    filters_clean = {k: v for k, v in filters.items()
                 if v is not None and k not in ("user_number", "grid", "random_start")
                 and not (k == "epr_fraction" and v == 1)}

    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    if not os.path.exists(base_folder):
        logger.warning(f"analysis_data folder not found: {base_folder}")
        return []

    full_file_map = {
        "stats":         "_stats.parquet",
        "rg_over_time":  "_rg_over_time.parquet",
        "s_over_time":   "_s_over_time.parquet",
        "venue_history": "_venue.parquet",
        "user_history":  "_history.parquet",
    }

    if needed_files is not None:
        file_map = {k: v for k, v in full_file_map.items() if k in needed_files}
        if not file_map:
            # Nothing per-user to load — caller will skip this function anyway,
            # but return an empty list gracefully just in case.
            return []
    else:
        file_map = full_file_map

    if needed_files:
        logger.info(f"Loading only: {sorted(needed_files)}")

    loaded = load_parquet_folder(base_folder, filters_clean, file_map, grid=grid,
                                 random_start=filters.get("random_start", False),
                                 user_number=user_number)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Analysis data loaded: {len(loaded)} users")
    return loaded


def _apply_user_number_slice(loaded, user_number):
    """
    Slice a list of loaded user entries according to the user_number filter.

    Positive user_number → keep first N.
    Negative user_number → keep last N.
    None or 0            → keep all.

    Note: for positive user_number the early-exit in load_parquet_folder already
    stops loading once N users are accumulated, so this function is typically a
    no-op in that case.  It is still called for safety and to handle the negative
    (last-N) case, which requires the full list.
    """
    if not user_number:   # covers None and 0 → keep all
        return loaded
    if user_number > 0:
        return loaded[:user_number]
    return loaded[user_number:]   # negative → last N


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_mobility_visualization(user_history, venue_history, stats = None, filters = None, title = None, show = False):
    logger.info("Plotting mobility visualization")
    """
    Visualizes the user mobility trajectory and visited venues.

    Parameters
    ----------
    user_history : pandas.DataFrame
        DataFrame containing the user's trajectory with columns including 'Destination_x_coord',
        'Destination_y_coord', and 'Clock'.
    venue_history : pandas.DataFrame
        DataFrame containing visited venues with columns 'Venue_x_coord', 'Venue_y_coord',
        and 'Number of visits'.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
        If provided, the center of mass and radius of gyration are plotted.
    title : str, optional
        Title for the plot.
    
    Returns
    -------
    None
    """


    x = venue_history["Venue_x_coord"].to_numpy()
    y = venue_history["Venue_y_coord"].to_numpy()
    num_visits = venue_history["Number of visits"].to_numpy()
    x_traj = user_history["Destination_x_coord"].to_numpy()
    y_traj = user_history["Destination_y_coord"].to_numpy()
    # Adding starting position
    x_traj, y_traj = np.insert(x_traj, 0, 0), np.insert(y_traj, 0, 0)

    _, ax = plt.subplots()

    # Normalize number of visits for colormap
    norm = plt.Normalize(num_visits.min(), num_visits.max())
    cmap = plt.cm.viridis  # Most used color scale
    sc = ax.scatter(x, y, s=10, c=num_visits, cmap=cmap, norm=norm, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Number of visits')
    ax.plot(x_traj, y_traj, color='red', linewidth=1, alpha=0.7)

    if stats is not None:
        r_gyr = stats.query('metric == "radius of gyration"')['value'].iloc[0]
        cm_x = stats.query('metric == "x_center of mass"')['value'].iloc[0]
        cm_y = stats.query('metric == "y_center of mass"')['value'].iloc[0]

        ax.scatter(cm_x,cm_y, marker='x', c="purple", alpha=0.6, s=50, zorder=3)
        circle = plt.Circle(
            (cm_x, cm_y),
            r_gyr,
            facecolor=(0.6, 0, 0.8, 0.2),
            edgecolor=(0.4, 0, 0.6, 0.8), 
            linewidth=2
        )
        ax.add_patch(circle) 

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title,fontsize=7)

    filepath_save_plot_or_show("plots", filters=filters, show=show)


def plot_rg_distribution(rg_list, filters = None, title=None, show=False):
    logger.info("Plotting rg distribution")
    """
    Plot the distribution of radius of gyration values on a log-log scale.

    This function creates a histogram of the radius of gyration (rg) values using
    logarithmic binning and plots the density on a log-log scale. It filters out
    zero values and uses a dotted histogram approach for better visualization.

    Parameters
    ----------
    rg_list : list or array-like
        List of radius of gyration values to plot.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """

    rg_array = np.array(rg_list)
    centers_filtered, density_filtered = dotted_histogram(rg_array)
     
    #counts_filtered = counts[mask]
    #widths_filtered = widths[mask]

    #print(f"\nNumero totale di punti: {len(rg_array)}")
    #print(f"Numero di bin con dati: {len(centers_filtered)}")
    #print("\nDettagli bin:")
    #for i in range(len(centers_filtered)):
        #print(f"Bin {i+1}: center={centers_filtered[i]:.2f}, "
          #f"counts={counts_filtered[i]}, "
          #f"width={widths_filtered[i]:.2f}, "
          #f"density={density_filtered[i]:.4f}")

    plt.loglog(centers_filtered, density_filtered, "o", markersize=5)

    plt.figure(figsize=(8, 6))
    plt.loglog(centers_filtered, density_filtered, "o", markersize=5)

    plt.xlabel("Radius of gyration")
    plt.ylabel("Density")
    if title:
        plt.title(title, fontsize=7)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    filepath_save_plot_or_show("plots", filters=filters, show=show)

def plot_frequency_distribution(venue_history_list, gamma, filters=None, title=None, show=False):

    logger.info("Plotting frequency distribution (averaged across users)")
    """
    Plot the rank-frequency distribution of venue visits averaged across all users.

    For each user the venues are ranked in descending order of visit count.
    The visit counts at each rank are then averaged across all users that have
    at least that many venues, producing a single mean curve.  A power-law fit
    line is overlaid for comparison.

    Parameters
    ----------
    venue_history_list : list of pandas.DataFrame or pandas.DataFrame
        One DataFrame per user, each containing a 'Number of visits' column.
        A bare DataFrame is accepted for backward compatibility (treated as a
        single-user list).
    gamma : float
        Exponent for the power-law fit line (f(x) = x^(-1-gamma)).
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """
    # Accept a single DataFrame for backward compatibility
    if isinstance(venue_history_list, pd.DataFrame):
        venue_history_list = [venue_history_list]

    # Collect visit counts per rank across all users
    rank_visits = {}   # rank (1-based) -> list of visit counts
    for vh in venue_history_list:
        sorted_visits = vh['Number of visits'].sort_values(ascending=False).values
        for rank, visits in enumerate(sorted_visits, start=1):
            rank_visits.setdefault(rank, []).append(visits)
            #created a dict where the key is the rank and the value is a list of visit counts for that rank across users

    ranks     = np.array(sorted(rank_visits.keys()))
    avg_visits = np.array([np.mean(rank_visits[r]) for r in ranks])

    plt.figure(figsize=(8, 6))
    plt.scatter(ranks, avg_visits, color='blue', label='avg data', s=50)

    x_fit = np.linspace(ranks.min(), ranks.max(), 100)
    y_fit = x_fit ** (-1 - gamma)
    y_fit = y_fit * avg_visits.max() / y_fit.max()   # scale to same maximum
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=r'$f(x) = x^{-1-\gamma}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel("Average frequency")
    if title:
        plt.title(title, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    filepath_save_plot_or_show("plots", filters=filters, show=show)

def plot_two_radius_vs_radius(rg_list, two_rg_list, filters = None, title = None, show = False):

    logger.info("Plotting 2-radius vs radius of gyration")
    """
    Plot the relationship between radius of gyration and 2-radius of gyration.

    This function creates a scatter plot comparing the standard radius of gyration
    with the 2-radius of gyration for a set of data points.

    Parameters
    ----------
    rg_list : list or array-like
        List of radius of gyration values.
    two_rg_list : list or array-like
        List of 2-radius of gyration values.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(rg_list, two_rg_list,s=5)
    plt.xlabel("Radius of Gyration")
    plt.ylabel("2-Radius of Gyration")
    if title:
        plt.title(title,fontsize=7)
    filepath_save_plot_or_show("plots", filters = filters, show=show)

def plot_S_over_time_from_df(s_over_time_list, filters=None, title=None, bins_per_decade=4, show=False):
    """
    Plot the radius of gyration over time as a percentage of each user's final
    (maximum) rg value, averaged across all users.

    For every log-spaced sample time T, each user contributes the rg value
    recorded at the last event with clock ≤ T (last-before rule), divided by
    their own maximum rg.  The per-user percentages are then averaged to obtain
    the mean fractional progress toward the final rg.

    Parameters
    ----------
    rg_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 'rg'].
    filters : dict, optional
        Used for the plot file name.
    title : str, optional
        Plot title.
    bins_per_decade : int, optional
        Number of log-spaced sample points per decade of time (default 4).
        The maximum clock value is always included as the last sample point.
    """
    logger.info("Plotting S over time (from analysis data)")

    if not s_over_time_list:
        logger.warning("No S-over-time data available to plot")
        return

    # Build log-spaced sample times spanning all users.
    all_clocks = np.concatenate([df["clock"].values for df in s_over_time_list])
    min_c, max_c = all_clocks.min(), all_clocks.max()
    log_min = np.log10(min_c)
    log_max = np.log10(max_c)
    num_samples = max(2, int(round((log_max - log_min) * bins_per_decade)))
    sample_times = np.logspace(log_min, log_max, num_samples)
    # Force exact max to avoid floating-point near-misses (10**log10(x) ≠ x)
    sample_times[-1] = max_c

    # For each sample time collect the normalised rg from every user
    pct_by_time = {t: [] for t in sample_times}

    for df in s_over_time_list:
        clocks = df["clock"].values
        s     = df["s"].values
        #max_s = s.max() Creating troubles!!!!!!!!!! s can decrease and max =\ from last value
        last_s = s[-1]
        if last_s == 0:
            logger.warning(f"Last s value is 0 for user in DataFrame")
            continue
        for t in sample_times:
            # Last record whose clock is ≤ t  (last-before rule)
            mask = clocks <= t
            if not mask.any():
                #0% for those who haven't started yet 
                pct_by_time[t].append(0.0)
            else:
                idx = int(np.where(mask)[0][-1])
                pct_by_time[t].append(s[idx])

    # Average across users for each sample time
    sample_t_plot = []
    avg_pct_plot  = []
    for t in sample_times:
        vals = pct_by_time[t]
        if vals:
            sample_t_plot.append(t)
            avg_pct_plot.append(np.mean(vals))

    plt.figure(figsize=(8, 5))
    plt.scatter(sample_t_plot, avg_pct_plot, marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Clock")
    plt.ylabel("S(t)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=7)

    filepath_save_plot_or_show(folder="plots", filters=filters, show=show)


def plot_r_gyr_over_time_from_df(rg_over_time_list, filters=None, title=None, bins_per_decade=4, show=False):
    """
    Plot the radius of gyration over time as a percentage of each user's final
    (maximum) rg value, averaged across all users.

    For every log-spaced sample time T, each user contributes the rg value
    recorded at the last event with clock ≤ T (last-before rule), divided by
    their own maximum rg.  The per-user percentages are then averaged to obtain
    the mean fractional progress toward the final rg.

    Parameters
    ----------
    rg_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 'rg'].
    filters : dict, optional
        Used for the plot file name.
    title : str, optional
        Plot title.
    bins_per_decade : int, optional
        Number of log-spaced sample points per decade of time (default 4).
        The maximum clock value is always included as the last sample point.
    """
    logger.info("Plotting r_gyr over time (normalised percentage, from analysis data)")

    if not rg_over_time_list:
        logger.warning("No rg-over-time data available to plot")
        return

    # Build log-spaced sample times spanning all users.
    all_clocks = np.concatenate([df["clock"].values for df in rg_over_time_list])
    min_c, max_c = all_clocks.min(), all_clocks.max()
    log_min = np.log10(min_c)
    log_max = np.log10(max_c)
    num_samples = max(2, int(round((log_max - log_min) * bins_per_decade)))
    sample_times = np.logspace(log_min, log_max, num_samples)
    # Force exact max to avoid floating-point near-misses (10**log10(x) ≠ x)
    sample_times[-1] = max_c

    # For each sample time collect the normalised rg from every user
    pct_by_time = {t: [] for t in sample_times}

    for df in rg_over_time_list:
        clocks = df["clock"].values
        rg     = df["rg"].values
        #max_rg = rg.max() Creating troubles!!!!!!!!!! rg can decrease and max =\ from last value
        last_rg = rg[-1]
        if last_rg == 0:
            logger.warning(f"Last rg value is 0 for user in DataFrame")
            continue
        for t in sample_times:
            # Last record whose clock is ≤ t  (last-before rule)
            mask = clocks <= t
            if not mask.any():
                #0% for those who haven't started yet 
                pct_by_time[t].append(0.0)
            else:
                idx = int(np.where(mask)[0][-1])
                pct_by_time[t].append(rg[idx] / last_rg * 100.0)

    # Average across users for each sample time
    sample_t_plot = []
    avg_pct_plot  = []
    for t in sample_times:
        vals = pct_by_time[t]
        if vals:
            sample_t_plot.append(t)
            avg_pct_plot.append(np.mean(vals))

    plt.figure(figsize=(8, 5))
    plt.scatter(sample_t_plot, avg_pct_plot, marker='o')
    plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Clock")
    plt.ylabel(r"$r_g(t)\,/\,r_g^{last}$ [%]")
    #plt.ylim(0, 105)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=7)

    filepath_save_plot_or_show(folder="plots", filters=filters, show=show)
    

def compute_and_save_aggregate_grid(analysis_folder):
    """
    Compute and save aggregate grid matrices at batch and parameter level.

    Batch level  : sums all user_i_grid.parquet inside analysis_folder and
                   saves the result as aggregate_grid.parquet in that same
                   folder.
    Param level  : re-scans every batch subfolder of the parent param folder,
                   sums their aggregate_grid.parquet files, and saves the
                   result as aggregate_grid.parquet directly in the param
                   folder (overwriting any previous version).

    Must be called after all per-user grid parquets have been written inside
    analysis_folder.

    Parameters
    ----------
    analysis_folder : str
        Path to the timestamped batch run folder produced by run_analysis()
        (e.g. analysis_data/ro_0.1_.../20260310_160442_960747/).
    """
    # --- batch-level aggregate ---
    grid_sum = None
    i = 0
    while True:
        fpath = os.path.join(analysis_folder, f"user_{i}_grid.parquet")
        if not os.path.exists(fpath):
            break
        mat = pd.read_parquet(fpath).values.astype(int)
        grid_sum = mat if grid_sum is None else grid_sum + mat
        i += 1

    if grid_sum is None:
        logger.warning(f"No user grid files found in {analysis_folder}; skipping aggregate.")
        return

    batch_agg = pd.DataFrame(grid_sum)
    batch_agg.columns = batch_agg.columns.astype(str)
    batch_agg.to_parquet(os.path.join(analysis_folder, "aggregate_grid.parquet"), index=False)
    logger.info(f"Batch-level aggregate grid saved to: {analysis_folder}")

    # --- param-level aggregate (sum over all existing batch aggregates) ---
    param_folder = os.path.dirname(analysis_folder)
    param_sum = None
    for entry in os.listdir(param_folder):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        agg_path = os.path.join(entry_path, "aggregate_grid.parquet")
        if not os.path.exists(agg_path):
            continue
        mat = pd.read_parquet(agg_path).values.astype(int)
        param_sum = mat if param_sum is None else param_sum + mat

    if param_sum is not None:
        param_agg = pd.DataFrame(param_sum)
        param_agg.columns = param_agg.columns.astype(str)
        param_agg.to_parquet(os.path.join(param_folder, "aggregate_grid.parquet"), index=False)
        logger.info(f"Param-level aggregate grid saved to: {param_folder}")


def load_all_aggregate_grids(filters):
    """
    Load param-level and all batch-level aggregate grids from analysis_data/.

    Parameters
    ----------
    filters : dict
        Simulation parameter filters to identify the param folder.
        Standard keys: "ro", "gamma", "a_param", "b_param", "time".
        Grid keys (when applicable): "cell_lenght", "cell_number".

    Returns
    -------
    param_grid : np.ndarray or None
        Param-level aggregate matrix (sum over all batches).
        None if the file does not exist.
    param_user_count : int
        Total number of users aggregated across all batches.
    batch_grids : list of (str, np.ndarray, int)
        One entry per batch subfolder that contains aggregate_grid.parquet,
        as (batch_id, grid_matrix, user_count) triples, sorted by batch_id.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    param_folder_name = (
        f"ro_{filters['ro']}_gamma_{filters['gamma']}"
        f"_a_{filters['a_param']}_b_{filters['b_param']}_time_{filters['time']}"
    )
    if "cell_lenght" in filters and "cell_number" in filters:
        param_folder_name += (
            f"_clen_{filters['cell_lenght']}_cnum_{filters['cell_number']}"
        )
        if filters.get("random_start", False):
            param_folder_name += "_rstart_yes"
    if filters.get("epr_fraction", 1.0) < 1.0:
        param_folder_name += f"_efrac_{filters['epr_fraction']}"

    param_folder = os.path.join(base_folder, param_folder_name)
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for aggregate grids: {param_folder}")
        return None, 0, []

    # Param-level aggregate
    param_agg_path = os.path.join(param_folder, "aggregate_grid.parquet")
    param_grid = (
        pd.read_parquet(param_agg_path).values.astype(int)
        if os.path.exists(param_agg_path) else None
    )

    # Batch-level aggregates — count user_*_grid.parquet files for each batch
    batch_grids = []
    param_user_count = 0
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        agg_path = os.path.join(entry_path, "aggregate_grid.parquet")
        if os.path.exists(agg_path):
            mat = pd.read_parquet(agg_path).values.astype(int)
            batch_user_count = sum(
                1 for f in os.listdir(entry_path)
                if f.startswith("user_") and f.endswith("_grid.parquet")
            )
            batch_grids.append((entry, mat, batch_user_count))
            param_user_count += batch_user_count

    logger.info(
        f"Loaded aggregate grids: 1 param-level ({param_user_count} users), "
        f"{len(batch_grids)} batch-level"
    )
    return param_grid, param_user_count, batch_grids


def load_all_batch_stats(filters):
    """
    Load batch_stats.parquet from every batch subfolder found inside the
    matching parameter folder in analysis_data/.

    Uses the same folder-naming convention as :func:`load_all_aggregate_grids`.

    Parameters
    ----------
    filters : dict
        Simulation parameter filters to identify the param folder.
        Standard keys: "ro", "gamma", "a_param", "b_param", "time".
        Grid keys (when applicable): "cell_lenght", "cell_number".
        Optional: "random_start" (bool), "epr_fraction" (float).

    Returns
    -------
    batch_stats_list : list of (str, pd.DataFrame)
        One entry per batch subfolder that contains batch_stats.parquet,
        as (batch_id, stats_df) pairs sorted by batch_id.
        Empty list when the param folder does not exist or no file is found.
    """
    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    param_folder_name = (
        f"ro_{filters['ro']}_gamma_{filters['gamma']}"
        f"_a_{filters['a_param']}_b_{filters['b_param']}_time_{filters['time']}"
    )
    if "cell_lenght" in filters and "cell_number" in filters:
        param_folder_name += (
            f"_clen_{filters['cell_lenght']}_cnum_{filters['cell_number']}"
        )
        if filters.get("random_start", False):
            param_folder_name += "_rstart_yes"
    if filters.get("epr_fraction", 1.0) < 1.0:
        param_folder_name += f"_efrac_{filters['epr_fraction']}"

    param_folder = os.path.join(base_folder, param_folder_name)
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for batch stats: {param_folder}")
        return []

    batch_stats_list = []
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        stats_path = os.path.join(entry_path, "batch_stats.parquet")
        if os.path.exists(stats_path):
            stats_df = pd.read_parquet(stats_path)
            batch_stats_list.append((entry, stats_df))

    logger.info(f"Loaded batch stats for {len(batch_stats_list)} batch(es)")
    return batch_stats_list


def print_batch_stats(batch_stats_list, filters=None, title=None):
    """
    Print a formatted summary table of batch-level statistics to stdout.

    Fully generic: iterates over every metric/value row present in each
    batch_stats.parquet without any hardcoded metric names.  This means
    that any new metric added to batch_stats.parquet during analysis is
    automatically displayed here without touching this function.

    One section is printed per batch, mirroring the per-batch loop used
    by the batch_grid plotter.  When more than one batch is available, a
    recap table follows with mean and std for every numeric metric.

    Parameters
    ----------
    batch_stats_list : list of (str, pd.DataFrame)
        As returned by :func:`load_all_batch_stats`:
        (batch_id, stats_df) pairs sorted by batch_id.
        Each DataFrame must have columns 'metric' and 'value'.
    filters : dict, optional
        Simulation parameters — printed as a header line for context.
    title : str, optional
        Optional title printed above the output block.
    """
    if not batch_stats_list:
        logger.warning("print_batch_stats: no batch stats to display.")
        return

    SEP_WIDE   = "=" * 72
    SEP_NARROW = "-" * 72

    # --- header ---
    print(f"\n{SEP_WIDE}")
    if title:
        print(f"  {title}")
    else:
        print("  BATCH STATS SUMMARY")
    if filters:
        param_str = "  params: " + ", ".join(f"{k}={v}" for k, v in filters.items())
        print(param_str)
    print(f"  {len(batch_stats_list)} batch(es) found")
    print(SEP_WIDE)

    # recap_data[metric_name] = [value_batch_0, value_batch_1, ...]
    recap_data = {}

    for batch_id, stats_df in batch_stats_list:
        print(f"\n  Batch : {batch_id}")
        print(SEP_NARROW)

        for _, row in stats_df.iterrows():
            metric = str(row["metric"])
            value  = row["value"]

            # Format integers without decimal point when the value is whole
            if isinstance(value, float) and value.is_integer():
                val_str = f"{int(value):>10d}"
            else:
                val_str = f"{value:>10.4f}"

            print(f"    {metric:<30s} {val_str}")
            recap_data.setdefault(metric, []).append(value)

        print(SEP_NARROW)

    # --- recap table (only when > 1 batch) ---
    if len(batch_stats_list) > 1:
        print(f"\n  {'RECAP':^70s}")
        print(SEP_WIDE)

        # Header: Batch | metric_0 | metric_1 | ...
        metric_names = list(recap_data.keys())
        col_w = 12
        hdr = f"  {'Batch':<26s}" + "".join(f"{m[:col_w]:>{col_w}s}" for m in metric_names)
        print(hdr)
        print(SEP_NARROW)

        # One row per batch
        for idx, (batch_id, stats_df) in enumerate(batch_stats_list):
            metrics_dict = dict(zip(stats_df["metric"], stats_df["value"]))
            row_str = f"  {batch_id:<26s}"
            for m in metric_names:
                v = metrics_dict.get(m, float("nan"))
                if isinstance(v, float) and v.is_integer():
                    row_str += f"{int(v):>{col_w}d}"
                else:
                    row_str += f"{v:>{col_w}.4f}"
            print(row_str)

        # Mean / std rows
        print(SEP_NARROW)
        for label, func in [("mean", np.mean), ("std", np.std)]:
            row_str = f"  {label:<26s}"
            for m in metric_names:
                vals = [v for v in recap_data[m] if not np.isnan(v)]
                row_str += f"{func(vals):>{col_w}.4f}" if vals else f"  {'n/a':>{col_w-2}s}"
            print(row_str)

        print(SEP_WIDE)

    print()

def plot_aggregate_grid(grid_matrix, filters=None, title=None, batch_id=None, mask_center=True, show=False):
    """ Plot a heatmap of an aggregate grid matrix.

        Parameters
        ----------
        grid_matrix : np.ndarray
            2D integer array with cumulative visit counts per cell.
        filters : dict, optional
            Used for the saved plot file name.
        title : str, optional
            Plot title.
        batch_id : str, optional
            When provided, included in the saved file name to distinguish individual batch plots from the param-level plot.
        mask_center : bool, optional
            Whether to mask the center cell in the plot.
            
    """

    logger.info(f"Plotting aggregate grid heatmap{' for batch ' + batch_id if batch_id else ''}")

    # Copy
    m = grid_matrix.copy()
    cx = m.shape[0] // 2
    cy = m.shape[1] // 2
    m_safe = np.where(m == 0, 1, m)
    if mask_center:
        m_masked = np.ma.array(m_safe, mask=False)
        m_masked.mask[cx, cy] = True
    else:
        m_masked = np.ma.array(m_safe, mask=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        m_masked.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        #norm=LogNorm()
    )

    plt.colorbar(im, ax=ax, label="Total visits")
    ax.set_xlabel("Cell X")
    ax.set_ylabel("Cell Y")

    if title:
        ax.set_title(title, fontsize=6)

    name = f"plot_aggregate_grid{'_' + batch_id if batch_id else ''}"

    filepath_save_plot_or_show("plots", name=name, filters=filters, show=show)