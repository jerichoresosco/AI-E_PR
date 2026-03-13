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


def filepath_save_plot(folder = "test_plots", name: str = None, filters: dict = None):
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

    # if name is not given, the calling function name is used instead
    if name is None:
        name = inspect.stack()[1].function

    # base folder 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # folder where plots will be saved
    plots_dir = os.path.join(base_dir, f"{folder}")

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
    parameters in a formatted string.

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
    if not filters:
        return base_title

    param_str = ", ".join(f"{k}={v}" for k, v in filters.items())
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

    mask = density > 0
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


# =============================================================================
# LOADERS
# =============================================================================

def load_parquet_folder(base_folder, filters, file_map, grid=False, return_run_folders=False):
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
    The "user_number" logic (first/last N) is handled by the callers
    load_raw_parquet() and load_analysis_parquet() before calling this function,
    so this function always loads all users that survive the filter.
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
        }
    elif grid == False:
        key_map = {
            "ro":           "ro",
            "gamma":        "gamma",
            "a_param":      "a",
            "b_param":      "b",
            "time":         "time",
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

        if any(converted.get(key_map.get(fk, fk)) != fv
               for fk, fv in filters.items()):
            continue

        for run_folder in os.listdir(folder_path):
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

    loaded = []
    for run_path, user_indices in matching_runs:
        for idx in user_indices:
            entry = {}
            for key, suffix in file_map.items():
                fpath = os.path.join(run_path, f"user_{idx}{suffix}")
                entry[key] = pd.read_parquet(fpath) if os.path.exists(fpath) else None
            if entry[next(iter(file_map))] is not None:
                if return_run_folders:
                    loaded.append({"data": entry, "run_folder": run_path})
                else:
                    loaded.append(entry)

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
                 if v is not None and k != "user_number" and k != "grid"}
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
                                 return_run_folders=return_run_folders)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Loaded {len(loaded)} users from raw_data")
    return loaded


def load_analysis_parquet(filters=None):
    """
    Load analysis data from analysis_data/.

    Parameters
    ----------
    filters : dict, optional
        Parameter filters. Accepted keys: "ro", "gamma", "a_param", "b_param",
        "time", "user_number".

    Returns
    -------
    list of dict
        One dict per user with keys: "user_history", "venue_history",
        "stats", "rg_over_time", "s_over_time".
        Any value may be None if the corresponding file is absent.
    """
    logger.info("Loading analysis data (analysis_data)...")

    if filters is None:
        filters = {}

    user_number = filters.get("user_number", None)
    filters_clean = {k: v for k, v in filters.items()
                     if v is not None and k != "user_number"}

    project_root = os.path.dirname(os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    ))
    base_folder = os.path.join(project_root, "analysis_data")

    if not os.path.exists(base_folder):
        logger.warning(f"analysis_data folder not found: {base_folder}")
        return []

    file_map = {
        "stats":         "_stats.parquet",
        "rg_over_time":  "_rg_over_time.parquet",
        "s_over_time":   "_s_over_time.parquet",
        "venue_history": "_venue.parquet",
        "user_history":  "_history.parquet",
    }

    loaded = load_parquet_folder(base_folder, filters_clean, file_map)
    loaded = _apply_user_number_slice(loaded, user_number)

    logger.info(f"Analysis data loaded: {len(loaded)} users")
    return loaded


def _apply_user_number_slice(loaded, user_number):
    """
    Slice a list of loaded user entries according to the user_number filter.

    Positive user_number → keep first N.
    Negative user_number → keep last N.
    None                 → keep all.
    """
    if user_number is None:
        return loaded
    if user_number >= 0:
        return loaded[:user_number]
    return loaded[user_number:]   # negative index slices from the end


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_mobility_visualization(user_history, venue_history, stats = None, filters = None, title = None):
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
        ax.set_title(title,fontsize=6)

    filepath_save_plot("plots", filters=filters)


def plot_rg_distribution(rg_list, filters = None, title=None):
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
        plt.title(title, fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    filepath_save_plot("plots", filters=filters)

def plot_frequency_distribution(venue_history, gamma, filters = None, title = None):   

    logger.info("Plotting frequency distribution")
    """
    Plot the frequency distribution of venue visits on a log-log scale.

    This function creates a rank-frequency plot of the number of visits per venue,
    sorted in descending order. It also overlays a power-law fit line for comparison.

    Parameters
    ----------
    venue_history : pandas.DataFrame
        DataFrame containing venue visit data with a 'Number of visits' column.
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

    freq = venue_history['Number of visits'].sort_values(ascending=False).reset_index(drop=True)
    x = freq.index.values + 1 #+1 in order to start from 1, so max value is first rank, not 0
    y = freq.values

    plt.figure(figsize=(8,6))

    plt.scatter(x, y, color='blue', label='data', s=50)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = x_fit**(-1-gamma)
    # scaling y so that function starts with the same maximum
    y_fit = y_fit * max(y)/max(y_fit)  
    plt.plot(x_fit, y_fit, color='red', linestyle='--', label=r'$f(x) = x^{-1-\gamma}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    if title:
        plt.title(title,fontsize=6)
    #plt.title("Rank-frequency plot of Number of visits per venue")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    filepath_save_plot("plots", filters=filters)

def plot_two_radius_vs_radius(rg_list, two_rg_list, filters = None, title = None):

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
        plt.title(title,fontsize=6)
    filepath_save_plot("plots", filters = filters)


def plot_S_over_time(user_history_list, filters = None, title = None, bins_per_decade=4):
    """
    Plot the exploration statistic S(t) over time on a log-log scale.

    This function computes and plots the cumulative number of unique venues visited
    (S(t)) as a function of time (clock), using logarithmic binning for averaging.

    Parameters
    ----------
    user_history_list : list of pandas.DataFrame
        List of DataFrames containing user history with 'Clock' and 'State(E/R)' columns.
    filters : dict, optional
        Dictionary of filters to be included in the plot title.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic binning.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """
    logger.info("Plotting S over time")
    plt.figure(figsize=(8,5))
    
    clocks = np.array([])
    s_values = np.array([])

    for df in user_history_list:
        e_rows = df[df['State(E/R)'] == 'E'].copy()
        
        clocks_temp = e_rows['Clock']
        s_values_temp = range(2, 2 + len(clocks_temp))

        clocks = np.append(clocks, clocks_temp)
        s_values = np.append(s_values, s_values_temp)

    min_clocks = clocks.min()
    max_clocks = clocks.max()

    log_min = np.log10(min_clocks)
    log_max = np.log10(max_clocks)

    #Avoiding boundary problems using a margin
    margin = 0.05
    log_min -= margin
    log_max += margin

    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))
   
    bins_edges = np.logspace(np.log10(min_clocks), np.log10(max_clocks), num_bins)
    y_mean, edges, _ = scipy.stats.binned_statistic(clocks, s_values, statistic='mean', bins=bins_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])      #geometric metric for log plot

    plt.figure(figsize=(7,5))

    plt.scatter(centers, y_mean, marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Clock")
    plt.ylabel("S(t)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=10)
    #plt.legend()
    
    filepath_save_plot(folder="plots", filters=filters)


def plot_S_over_time_from_df(s_over_time_list, filters=None, title=None, bins_per_decade=4):
    """
    Plot S(t) — cumulative distinct venues visited — over time using
    pre-computed s_over_time DataFrames from analysis_data/.

    Parameters
    ----------
    s_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 's'].
    filters : dict, optional
        Used for the plot file name.
    title : str, optional
        Plot title.
    bins_per_decade : int, optional
        Log-axis binning density.
    """
    logger.info("Plotting S over time (from analysis data)")

    if not s_over_time_list:
        logger.warning("No S-over-time data available to plot")
        return

    clocks   = np.concatenate([df["clock"].values for df in s_over_time_list])
    s_values = np.concatenate([df["s"].values     for df in s_over_time_list])

    min_c, max_c = clocks.min(), clocks.max()
    log_min  = np.log10(min_c) - 0.05
    log_max  = np.log10(max_c) + 0.05
    num_bins = int(np.ceil((log_max - log_min) * bins_per_decade))
    bin_edges = np.logspace(log_min, log_max, num_bins)

    import scipy.stats
    y_mean, edges, _ = scipy.stats.binned_statistic(
        clocks, s_values, statistic='mean', bins=bin_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    plt.figure(figsize=(8, 5))
    plt.scatter(centers, y_mean, marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Clock")
    plt.ylabel("S(t)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=10)

    filepath_save_plot(folder="plots", filters=filters)


def plot_r_gyr_over_time_from_df(rg_over_time_list, filters=None, title=None, bins_per_decade=4):
    """
    Plot r_gyr(t) — radius of gyration as a function of time — using
    pre-computed rg_over_time DataFrames from analysis_data/.

    Parameters
    ----------
    rg_over_time_list : list of pd.DataFrame
        Each DataFrame has columns ['clock', 'rg'].
    filters : dict, optional
        Used for the plot file name.
    title : str, optional
        Plot title.
    bins_per_decade : int, optional
        Log-axis binning density.
    """
    logger.info("Plotting r_gyr over time (from analysis data)")

    if not rg_over_time_list:
        logger.warning("No rg-over-time data available to plot")
        return

    clocks    = np.concatenate([df["clock"].values for df in rg_over_time_list])
    rg_values = np.concatenate([df["rg"].values    for df in rg_over_time_list])

    min_c, max_c = clocks.min(), clocks.max()
    log_min  = np.log10(min_c) - 0.05
    log_max  = np.log10(max_c) + 0.05
    num_bins = int(np.ceil((log_max - log_min) * bins_per_decade))
    bin_edges = np.logspace(log_min, log_max, num_bins)

    import scipy.stats
    rg_mean, edges, _ = scipy.stats.binned_statistic(
        clocks, rg_values, statistic='mean', bins=bin_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    plt.figure(figsize=(8, 5))
    plt.scatter(centers, rg_mean, marker='o')
    plt.xscale("log")
    plt.xlabel("Clock")
    plt.ylabel("r_gyr(t)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=10)

    filepath_save_plot(folder="plots", filters=filters)
    

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
    batch_grids : list of (str, np.ndarray)
        One entry per batch subfolder that contains aggregate_grid.parquet,
        as (batch_id, grid_matrix) pairs, sorted by batch_id.
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

    param_folder = os.path.join(base_folder, param_folder_name)
    if not os.path.exists(param_folder):
        logger.warning(f"Param folder not found for aggregate grids: {param_folder}")
        return None, []

    # Param-level aggregate
    param_agg_path = os.path.join(param_folder, "aggregate_grid.parquet")
    param_grid = (
        pd.read_parquet(param_agg_path).values.astype(int)
        if os.path.exists(param_agg_path) else None
    )

    # Batch-level aggregates
    batch_grids = []
    for entry in sorted(os.listdir(param_folder)):
        entry_path = os.path.join(param_folder, entry)
        if not os.path.isdir(entry_path):
            continue
        agg_path = os.path.join(entry_path, "aggregate_grid.parquet")
        if os.path.exists(agg_path):
            mat = pd.read_parquet(agg_path).values.astype(int)
            batch_grids.append((entry, mat))

    logger.info(
        f"Loaded aggregate grids: 1 param-level, {len(batch_grids)} batch-level"
    )
    return param_grid, batch_grids


def plot_aggregate_grid(grid_matrix, filters=None, title=None, batch_id=None):
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
            
    """

    logger.info(f"Plotting aggregate grid heatmap{' for batch ' + batch_id if batch_id else ''}")

    # Copy
    m = grid_matrix.copy()
    cx = m.shape[0] // 2
    cy = m.shape[1] // 2
    m_safe = np.where(m == 0, 1, m)
    m_masked = np.ma.array(m_safe, mask=False)
    m_masked.mask[cx, cy] = True
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        m_masked.T,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        norm=LogNorm()
    )

    plt.colorbar(im, ax=ax, label="Total visits")
    ax.set_xlabel("Cell X")
    ax.set_ylabel("Cell Y")

    if title:
        ax.set_title(title, fontsize=8)

    name = f"plot_aggregate_grid{'_' + batch_id if batch_id else ''}"

    filepath_save_plot("plots", name=name, filters=filters)