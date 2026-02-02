import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import inspect
import model.user
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


def filepath_save_plot(folder = "test_plots", name: str = None):
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

    
    filename = f"{name}.pdf"
    filepath = os.path.join(plots_dir, filename)
    
    plt.savefig(filepath)
    plt.close()

def plot_mobility_visualization(user_history, venue_history, stats = None, sizes = 20, rg=False, title = None):
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
    stats : pandas.DataFrame, optional
        DataFrame containing user statistics, e.g., radius of gyration and center of mass.
        If provided, the center of mass and radius of gyration are plotted.
    sizes : int, default=20
        Scaling factor for marker sizes representing venue visit frequency.
    rg : bool, optional
        If True, indicates whether to plot radius of gyration (default: False).
    title : str, optional
        Title for the plot.
    
    Returns
    -------
    None
    """


    x,y = venue_history["Venue_x_coord"].to_numpy(), venue_history["Venue_y_coord"].to_numpy()
    marker_sizes = venue_history["Number of visits"].to_numpy() * sizes
    x_traj, y_traj = user_history["Destination_x_coord"].to_numpy(), user_history["Destination_y_coord"].to_numpy()
    #Adding starting position
    x_traj, y_traj = np.insert(x_traj, 0, 0), np.insert(y_traj, 0, 0)
    
    _, ax = plt.subplots()  

    ax.scatter(x, y, s=marker_sizes, marker='x', c='blue', alpha=0.6)
    ax.plot(x_traj, y_traj, color='red', linewidth=1, alpha=0.7)

    if stats is not None:
        r_gyr = stats.query('metric == "radius of gyration"')['value'].iloc[0]
        cm_x = stats.query('metric == "x_center of mass"')['value'].iloc[0]
        cm_y = stats.query('metric == "y_center of mass"')['value'].iloc[0]

        ax.scatter(cm_x,cm_y, c="purple", alpha=0.6, s=50, zorder=3)
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

    filepath_save_plot("plots")


def plot_rg_distribution(rg_list, bins_per_decade=4, title=None):
    """
    Plot the distribution of radius of gyration values on a log-log scale.

    This function creates a histogram of the radius of gyration (rg) values using
    logarithmic binning and plots the density on a log-log scale. It filters out
    zero values and uses a dotted histogram approach for better visualization.

    Parameters
    ----------
    rg_list : list or array-like
        List of radius of gyration values to plot.
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic binning.
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
    filepath_save_plot("plots")

def plot_frequency_distribution(venue_history, gamma, title = None):   
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
    filepath_save_plot("plots")

def plot_two_radius_vs_radius(rg_list, two_rg_list, title = None):
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
    filepath_save_plot("plots")


def build_plot_title(base_title, sim_params):
    """
    Build a plot title by combining a base title with simulation parameters.

    This function constructs a descriptive title for plots by appending simulation
    parameters in a formatted string.

    Parameters
    ----------
    base_title : str
        The base title for the plot.
    sim_params : dict
        Dictionary of simulation parameters to include in the title.

    Returns
    -------
    str
        The constructed title string.
    """
    if not sim_params:
        return base_title

    param_str = ", ".join(f"{k}={v}" for k, v in sim_params.items())
    return f"{base_title} ({param_str})"
        

def polar_to_cartesian (delta_r, angle):
    """
    Convert polar coordinates to Cartesian coordinates.

    This function transforms a distance and angle in polar coordinates
    to x and y coordinates in Cartesian space.

    Parameters
    ----------
    delta_r : float
        The radial distance from the origin.
    angle : float
        The angle in radians from the positive x-axis.

    Returns
    -------
    tuple of float
        A tuple (x, y) representing the Cartesian coordinates.
    """
    x = delta_r*np.cos(angle)
    y = delta_r*np.sin(angle)
    return (x,y)


def load_simulations_parquet(filters=None, base_folder=None):
    """
    Load simulation data from Parquet files based on parameter filters.

    This function scans a directory structure for simulation data stored in Parquet format,
    applying filters to select specific simulation runs. It loads user history, venue history,
    and optional statistics for each matching simulation.

    Parameters
    ----------
    filters : dict, optional (default=None)
        Dictionary of filters to apply, e.g., {"ro": 0.6, "b_param": 0.8, "time": 100}.
        Keys should match parameter names; None values are ignored.
    base_folder : str, optional (default=None)
        Root folder containing simulation data. Defaults to "Data" in the project root.

    Returns
    -------
    list of tuples
        List of tuples, each containing (user_history_df, venue_history_df, stats_df).
        stats_df may be None if not available.
    """

    # Default: empty filters
    if filters is None:
        filters = {}

    # Remove filters with None (ignore them)
    filters = {k: v for k, v in filters.items() if v is not None}

    # Default base folder
    if base_folder is None:
        project_root = os.path.dirname(os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        ))
        base_folder = os.path.join(project_root, "Data")

    if not os.path.exists(base_folder):
        return []

    loaded_users = []

    # Map filter keys to folder keys
    key_map = {
        "a_param": "a",
        "b_param": "b",
        "ro": "ro",
        "gamma": "gamma",
        "time": "time"
    }

    # Iterate over simulation folders
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse folder name
        parts = folder.split("_")
        if len(parts) % 2 != 0:
            continue  # malformed

        params = {}
        try:
            for i in range(0, len(parts), 2):
                k, v = parts[i], parts[i + 1]
                params[k] = v
        except Exception:
            continue

        # Convert params to Python types
        converted = {}
        for k, v in params.items():
            if k == "time":
                converted[k] = int(v)
            else:
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v  # fallback

        # Apply filters
        match = True
        for f_key, f_value in filters.items():
            folder_key = key_map[f_key]
            if converted.get(folder_key) != f_value:
                match = False
                break

        if not match:
            continue

        # Load data inside the run folders
        for run_folder in os.listdir(folder_path):
            run_path = os.path.join(folder_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            # Gather all users in this run
            users_in_run = {}

            for file in os.listdir(run_path):
                full = os.path.join(run_path, file)

                if file.endswith("_history.parquet"):
                    idx = file.split("_")[1]
                    users_in_run.setdefault(idx, {})["user_history"] = pd.read_parquet(full)

                elif file.endswith("_venue.parquet"):
                    idx = file.split("_")[1]
                    users_in_run.setdefault(idx, {})["venue_history"] = pd.read_parquet(full)

                elif file.endswith("_stats.parquet"):
                    idx = file.split("_")[1]
                    users_in_run.setdefault(idx, {})["stats"] = pd.read_parquet(full)

            # Append valid users
            for d in users_in_run.values():
                uh = d.get("user_history")
                vh = d.get("venue_history")
                st = d.get("stats", None)
                if uh is not None and vh is not None:
                    loaded_users.append((uh, vh, st))

    return loaded_users

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

def plot_S_over_time(user_history_list, title = None, bins_per_decade=4, k = None):
    """
    Plot the exploration statistic S(t) over time on a log-log scale.

    This function computes and plots the cumulative number of unique venues visited
    (S(t)) as a function of time (clock), using logarithmic binning for averaging.

    Parameters
    ----------
    user_history_list : list of pandas.DataFrame
        List of DataFrames containing user history with 'Clock' and 'State(E/R)' columns.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic binning.
    k : int, optional (default=None)
        Number of user histories to include. If None, include all.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """
    if k is not None:
        dfs_to_plot = user_history_list[:k]
    else:
        dfs_to_plot = user_history_list

    plt.figure(figsize=(8,5))
    
    clocks = np.array([])
    s_values = np.array([])

    for df in dfs_to_plot:
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
    
    filepath_save_plot(folder="plots")



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


def plot_r_gyr_over_time(user_history_list, title=None, bins_per_decade=4, k=None):
    """
    Plot the cumulative radius of gyration r_gyr(t) over time on a log scale.

    This function computes the radius of gyration for each time step in the user histories
    and plots the mean r_gyr(t) using logarithmic binning.

    Parameters
    ----------
    user_history_list : list of pandas.DataFrame
        List of DataFrames containing user history with 'Clock', 'Destination_x_coord',
        and 'Destination_y_coord' columns.
    title : str, optional (default=None)
        Title for the plot. If None, no title is set.
    bins_per_decade : int, optional (default=4)
        Number of bins per decade for logarithmic binning.
    k : int, optional (default=None)
        Number of user histories to include. If None, include all.

    Returns
    -------
    None
        The function generates and saves a plot but does not return any value.
    """
    
    if k is not None:
        dfs_to_plot = user_history_list[-k:]
    else:
        dfs_to_plot = user_history_list

    clocks = np.array([])
    r_gyr_values = np.array([])

    for df in dfs_to_plot:
        clocks_temp = df['Clock'].values  
        r_gyr_temp = []
        for i in range(len(df)):
            # seleziona tutte le righe fino alla riga corrente inclusa
            traj_x = np.concatenate(([0], df.iloc[:i+1]['Destination_x_coord'].values))
            traj_y = np.concatenate(([0], df.iloc[:i+1]['Destination_y_coord'].values))

            x_cm = np.mean(traj_x)
            y_cm = np.mean(traj_y)
            r_gyr_val = np.sqrt(np.mean((traj_x - x_cm)**2 + (traj_y - y_cm)**2))
            r_gyr_temp.append(r_gyr_val)

        clocks = np.append(clocks, clocks_temp)
        r_gyr_values = np.append(r_gyr_values, r_gyr_temp)

    # bin logaritmici
    min_clocks = clocks.min()
    max_clocks = clocks.max()

    log_min = np.log10(min_clocks) - 0.05
    log_max = np.log10(max_clocks) + 0.05
    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))
    bins_edges = np.logspace(log_min, log_max, num_bins)

    r_gyr_mean, edges, _ = scipy.stats.binned_statistic(clocks, r_gyr_values, statistic='mean', bins=bins_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])  # centro geometrico per bin log

    # plot
    plt.figure(figsize=(8,5))
    plt.scatter(centers, r_gyr_mean, marker='o')
    plt.xscale("log")
    # plt.yscale("log")  # opzionale
    plt.xlabel("Clock")
    plt.ylabel("Raggio di girazione r_gyr(t)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    if title:
        plt.title(title, fontsize=10)
    
    plt.tight_layout()
    filepath_save_plot("plots")

