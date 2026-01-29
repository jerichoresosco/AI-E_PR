import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import inspect
import model.user

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

def temporal_power_law_sampling( par = 0.8, min_val = 300, max_val = 61200, n_samples = 1):    
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


def compute_exploration_points(user, starting_point=None, x_range=(-10, 10), y_range=(-10, 10), marker_scale=20, k_top = None):
    """
    Compute the 2D coordinates for exploration steps of a User based on angle and distance (delta_r),
    and size the markers according to the number of visits in venue_history.

    Parameters
    ----------
    user : User
        An instance of the User class containing user_history and venue_history as DataFrames.
    starting_point : tuple, optional
        (x0, y0) coordinates to start from. If None, a random starting point is generated
        within the provided x_range and y_range.
    x_range : tuple of float, default (-10, 10)
        Range for random X coordinate if starting_point is None.
    y_range : tuple of float, default (-10, 10)
        Range for random Y coordinate if starting_point is None.
    marker_scale : float, default 20
        Scaling factor for marker sizes according to number of visits.

    Returns
    -------
    x : np.ndarray
        X coordinates of exploration points.
    y : np.ndarray
        Y coordinates of exploration points.
    marker_sizes : np.ndarray
        Marker sizes proportional to the number of visits for each venue.
    """

    # Determine starting point
    if starting_point is None:
        x0 = np.random.uniform(*x_range)
        y0 = np.random.uniform(*y_range)
    else:
        x0, y0 = starting_point

    # Filter only exploration steps
    explore_df = user.user_history[user.user_history["State(E/R)"] == "E"]

    if explore_df.empty:
        # No exploration steps
        logger.warning("No exploration is taking place! Impossible to compute exploration points")
        return np.array([]), np.array([]), np.array([])

    # Compute X and Y coordinates using polar to Cartesian conversion
    x_venue, y_venue = np.array([x0]),np.array([y0])
    for row in explore_df.itertuples():

        x = x_venue[row.Starting_Venue - 1] + row.Distance * np.cos(row.Angle) #-1 because starting_venue[0]=1
        y = y_venue[row.Starting_Venue - 1] + row.Distance * np.sin(row.Angle) # and not = 0
        x_venue = np.append(x_venue, x)
        y_venue = np.append(y_venue, y)

    # Map number of visits from venue_history
    number_visits = user.venue_history["Number of visits"].to_numpy()

    # Marker sizes proportional to number of visits
    marker_sizes = number_visits * marker_scale

    x_traj = np.array([])
    y_traj = np.array([])

    if k_top is not None:
        # Get the top-k venues by number of visits
        top_k_venues = (
            user.venue_history
            .sort_values("Number of visits", ascending=False)
            .head(k_top)["Venue"]
            .to_numpy()
            )
    else:
        top_k_venues = None
        
    trajectory = trajectory_from_user(user)
    for position in (trajectory):
        if (top_k_venues is None) or (position in top_k_venues):
            index = position - 1  #x_venue 0-th element is 1
            x_traj = np.append(x_traj, x_venue[index])
            y_traj = np.append(y_traj, y_venue[index])

    return x_venue, y_venue, marker_sizes, x_traj, y_traj


import matplotlib.pyplot as plt

def plot_mobility_visualization(user_history, venue_history, stats = None, sizes = 20, rg=False, title = None):
    """Plots places visited by the users in the space with trajectories"""


    x,y = venue_history["Venue_x_coord"].to_numpy(), venue_history["Venue_y_coord"].to_numpy()
    marker_sizes = venue_history["Number of visits"].to_numpy() * sizes
    x_traj, y_traj = user_history["Destination_x_coord"].to_numpy(), user_history["Destination_y_coord"].to_numpy()
    #Adding starting position
    x_traj, y_traj = np.insert(x_traj, 0, 0), np.insert(y_traj, 0, 0)
    
    _, ax = plt.subplots()  

    ax.scatter(x, y, s=marker_sizes, c='blue', alpha=0.6)
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

    
def trajectory_from_user(user):

    """
    Calculate the trajectory of a user

    Parameters:
    -----------
    user: the user whom calculation is needed (User)
     
    Returns:
    --------
    the trajectory of the user (Tuple(np.array, np.array)) """

    trajectory_without_starting_position = user.user_history["Destination"].to_numpy()
    trajectory = np.insert(trajectory_without_starting_position, 0, 1) #Adding 1 because 1 is starting position
    

    return trajectory


def convert_histories_to_df(user):
        """
        Convert the user history and venue history lists into Pandas DataFrames.        

        """
        #User_history
        columns_user = user.user_history[0]   
        data_user = user.user_history[1:]     
        user.user_history = pd.DataFrame(data_user, columns=columns_user)

        #venue_history
        columns_venue = user.venue_history[0]
        data_venue = user.venue_history[1:]    
        user.venue_history = pd.DataFrame(data_venue, columns=columns_venue)


def plot_rg_distribution(rg_list, bins_per_decade=4, title=None):
    """
    Showing radius of gyration distribution
    """

    rg_array = np.array(rg_list)
    rg_array = rg_array[rg_array > 0]   # evita zeri

    min_rg = rg_array.min()
    max_rg = rg_array.max()

    log_min = np.log10(min_rg)
    log_max = np.log10(max_rg)

    #Avoiding boundary problems using a margin
    margin = 0.05
    log_min -= margin
    log_max += margin

    num_decades = log_max - log_min
    num_bins = int(np.ceil(num_decades * bins_per_decade))

    # Creation of logarithmic bins
    bins = np.logspace(log_min, log_max, num_bins + 1)

    # frequencies
    counts, edges = np.histogram(rg_array, bins=bins)

    # bin center
    centers = np.sqrt(edges[:-1] * edges[1:])

    # normailization
    widths = np.diff(edges)
    density = counts / widths / len(rg_array)

    mask = density > 0
    centers_filtered = centers[mask]
    density_filtered = density[mask]
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

    plt.figure(figsize=(8, 6))
    plt.scatter(rg_list, two_rg_list)
    plt.xlabel("Radius of Gyration")
    plt.ylabel("2-Radius of Gyration")
    if title:
        plt.title(title,fontsize=6)
    filepath_save_plot("plots")


def build_plot_title(base_title, sim_params):
    """
    Create a plot title using a base title and a dictionary of simulation parameters.
    """
    if not sim_params:
        return base_title

    param_str = ", ".join(f"{k}={v}" for k, v in sim_params.items())
    return f"{base_title} ({param_str})"
        

def polar_to_cartesian (delta_r, angle):
    x = delta_r*np.cos(angle)
    y = delta_r*np.sin(angle)
    return (x,y)


def load_simulations_parquet(filters=None, base_folder=None):
    """
    Load simulation data from Parquet based on parameter filters.

    Args:
        filters (dict): dict of filters, e.g.
                        {"ro": 0.6, "b_param": 0.8, "time": 100}
                        Values = None → ignored.
        base_folder (str): root folder, default = Data at project root.

    Returns:
        List of tuples: (user_history_df, venue_history_df, stats_df or None)
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
    Extract the trajectory coordinates from user_history as NumPy arrays,
    optionally filtering only the top-k most visited venues.

    Parameters
    ----------
    user : User
        An object with user_history and venue_history as lists of lists.
    k_top : int or None
        Number of top visited venues to include. If None, include all.

    Returns
    -------
    x_traj : np.ndarray
        X coordinates of the trajectory (starts with 0).
    y_traj : np.ndarray
        Y coordinates of the trajectory (starts with 0).
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
