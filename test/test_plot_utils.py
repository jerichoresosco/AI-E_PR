import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import model.utils
import os

def test_power_law_sampling():
    """
    Generate and visualize samples drawn from the power-law distribution
    implemented in `power_law_sampling()`, and compare them with the
    theoretical probability density function.

    The function performs the following steps:
    1. Generates `n_samples` values using inverse transform sampling.
    2. Plots the histogram of the sampled values on a log-log scale.
    3. Computes the analytical power-law distribution p(x) for comparison.
    4. Plots both the theoretical curve and the sampled histogram to visually
       assess the agreement between the model and the generated data.

    Parameters:
    ----------
    None

    Returns:
    --------
    None
        This function only produces plots for visual inspection and does not
        return any value.
    """

    #Using values as in the spatial case
    par = 0.55
    min_val = 1
    max_val = 100
    n_samples = 100000

    # Samples generation
    samples = model.utils.power_law_sampling(par=par, min_val=min_val, max_val=max_val,n_samples=n_samples)

    #Plotting results
    plt.figure(figsize=(12, 5))

    # histogam of samples
    plt.subplot(1, 2, 1)
    plt.hist(samples, bins=90, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('r')
    plt.ylabel('Probability density')
    plt.title(f"Samples histogram (par = {par})")
    plt.yscale('log')
    plt.xscale('log')

    # subplot of theorical p(r) vs generated samples
    plt.subplot(1, 2, 2)
    r_theory = np.linspace(min_val, max_val, 1000)
    #Normalization
    norm_const = (-par) / (max_val**(-par) - min_val**(-par))
    p_theory = norm_const * r_theory**(-1-par)

    plt.plot(r_theory, p_theory, 'r-', linewidth=2, label='Theoretical p(x)')
    plt.hist(samples, bins=90, density=True, alpha=0.5, label='Samples')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Theoretical vs Samples')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    plt.tight_layout()

    model.utils.filepath_save_plot(folder="test_plots")

    plt.close()


test_power_law_sampling()