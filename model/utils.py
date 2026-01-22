import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import inspect

import logging
logger = logging.getLogger(__name__)

def power_law_sampling( par, min_val, max_val, n_samples):
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


def spatial_power_law_sampling( par = 0.55, min_val = 5, max_val = 100000, n_samples = 1):    
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


def create_venue_matrix_from_users(users, fill_value=0):
    """
    Creates a matrix (pandas) with venues x users structure reading
    directly from User's list 
        
    Parameters:
    ----------
    users (list): list of User istances
    fill_value: filler-value for empty elements(default: np.nan)
       
    Returns:
    --------
    pd.DataFrame: matrix with Venue_i (rows) e User_j (columns)
    """
        
    user_visits = []
    for user in users:
        visits = [row[1] for row in user.venue_history[1:]]
        user_visits.append(visits)
       
    max_len = max(len(visits) for visits in user_visits)
        
    data = {}
    for i, visits in enumerate(user_visits):
        user_id = f"{users[i].uid}"
        padded = visits + [fill_value] * (max_len - len(visits))
        data[user_id] = padded
        
    df = pd.DataFrame(data, index=[f"Venue_{i+1}" for i in range(max_len)])
        
    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_exploration_points(user, starting_point=None, x_range=(-10, 10), y_range=(-10, 10), marker_scale=20):
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

    trajectory = trajectory_from_user(user)
    for position in (trajectory):
        index = position - 1  #x_venue 0-th element is 1
        x_traj = np.append(x_traj, x_venue[index])
        y_traj = np.append(y_traj, y_venue[index])

    return x_venue, y_venue, marker_sizes, x_traj, y_traj


import matplotlib.pyplot as plt

def plot_mobility_visualization(x, y, sizes, x_traj, y_traj, rg=False):
    """Plots places visited by the users in the space with trajectories"""
    
    _, ax = plt.subplots()  

    ax.scatter(x, y, s=sizes, c='blue', alpha=0.6)
    ax.plot(x_traj, y_traj, color='red', linewidth=1, alpha=0.7)

    if rg:
        r_cm, r_gyr = radius_of_gyration(x_traj, y_traj)

        ax.scatter(*r_cm, c="purple", alpha=0.6, s=50, zorder=3)
        circle = plt.Circle(
            tuple(r_cm),
            r_gyr,
            facecolor=(0.6, 0, 0.8, 0.2),
            edgecolor=(0.4, 0, 0.6, 0.8), 
            linewidth=2
        )
        ax.add_patch(circle) 

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Mobility")

    plt.show()

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

def radius_of_gyration(x_traj, y_traj):
    """
    Calculates the radius of gyration and the center of mass given the trajectory of user as two distinct 
    numpy vectors containing coordinates
    
    :param x_traj: x coordinates of trajectory
    :param y_traj: y coordinates of trajectory

    Returns:
    --------
    radius of gyration (float)
    """
    x_cm = np.mean(x_traj)
    y_cm = np.mean(y_traj)

    r_cm = np.array([x_cm, y_cm])
    r_gyr = np.sqrt(np.mean((x_traj - x_cm)**2 + (y_traj - y_cm)**2))

    return r_cm, r_gyr
