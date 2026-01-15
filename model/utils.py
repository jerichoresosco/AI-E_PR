import numpy as np
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

    logger.info(
        f"Generating {n_samples} samples from power-law "
        f"distribution (par={par}, min_val={min_val}, max_val={max_val})"
    )


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
    logger.info("Spatial sampling...")
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
    logger.info("Temporal sampling...")
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
