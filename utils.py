import numpy as np
import matplotlib.pyplot as plt

def power_law_sampling(x_min=5, x_max=100000, a = 0.55):
    """
    Sampling from analytical distribution for distancies as done in the EPR paper
    "Modelling the scaling properties of  human mobility"
    
    Parameters:
    ----------
    x_min : float
        Has to be >0 in order to avoid numerical problems (normalization), chosen 5m 
        arbitrarly as a unit-base for movement
    x_max : float
        Chosen 100km as in the paper
    a : float
        Chosen 0.55 as in the paper. Cant be 
        
    
    Return:
    --------
    number : float
        Returns the number sampled from the power law distribution
    """
        
    # Genera un numero casuale uniforme tra 0 e 1
    u = np.random.uniform(0, 1)
 