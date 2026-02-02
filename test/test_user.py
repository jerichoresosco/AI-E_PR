import unittest
import model.user
import model.utils
import model.simulation
import pandas as pd
import pprint
import copy
from matplotlib import pyplot as plt
import numpy as np

class TestUser(unittest.TestCase):
    
    ##THIS SUPER METHOD IS INITIALIZING THE SIMULATION
    @classmethod
    def setUpClass(cls):
        cls.master_simulation = model.simulation.Simulation(
            ro=0.6, 
            gamma=-0.21, 
            a_param=0.55, 
            b_param=0.8, 
            user_number=1, 
            time=5*17*3600
        )
        cls.master_simulation.run()

    ##THIS METHOD ASSIGNS TO EACH METHOD OF TESTUSER CLASS A COPY OF
    ## THE MASTER SIMULATION, SO THAT CHANGES DURING TESTING WILL NOT SHARED
    ## AMONG DIFFERENT METHODS
    def setUp(self):
        self.simulation = copy.deepcopy(self.master_simulation)

    def test_user_history_type(self):
        """Test if the type of user_history after .run is a list"""
        self.assertIsInstance(self.simulation.user.user_history, list)

    def test_venue_history_type(self):
        """Test if the type of venue_history after .run is a list"""
        self.assertIsInstance(self.simulation.user.venue_history, list)

    def test_trajectory_continuity(self):

        """Test if there is continuity between positions of user"""
        user_history = self.simulation.user.user_history
        
        for i in range(1, len(user_history) - 1):

            #The test funztion is assuming the static position of columns in user_history
            #It seems to me an excess here the retrive of the index.
            current_destination = user_history[i][4]
            current_dest_x = user_history[i][5]
            current_dest_y = user_history[i][6]
            
            next_departure = user_history[i + 1][1]
            next_dep_x = user_history[i + 1][2]
            next_dep_y = user_history[i + 1][3]
            
            self.assertEqual(current_destination, next_departure,
                            f"Discontinuità nel nome della posizione alla riga {i}: "
                            f"destinazione '{current_destination}' != partenza successiva '{next_departure}'")
            self.assertEqual(current_dest_x, next_dep_x,
                            f"Discontinuità nella coordinata X alla riga {i}: "
                            f"{current_dest_x} != {next_dep_x}")
            self.assertEqual(current_dest_y, next_dep_y,
                            f"Discontinuità nella coordinata Y alla riga {i}: "
                            f"{current_dest_y} != {next_dep_y}")

    def test_get_return_position(self):
        """
        Statistical test for get_return_position().

        The theoretical probability is computed analytically for ALL venues,
        assigning probability 0 to the current venue (the one at position (5,5)).
        """

        user = self.simulation.user

        # Override venue history manually
        user.venue_history = [
            ["Venue_name", "Number of visits", "Venue_x_coord", "Venue_y_coord"],
            ["Venue_1", 1, 0, 0],
            ["Venue_2", 2, 1, 1],
            ["Venue_3", 3, 2, 2],
            ["Venue_4", 4, 3, 3],
            ["Venue_5", 5, 4, 4],
            ["Venue_6", 6, 5, 5],   # current position
        ]

        user._venue_index = {f"Venue_{i}": i for i in range(1,7)}

        # Current position = Venue_6, must have probability 0
        user.position = (5,5)

        # ALL venues theoretical probability (Venue_6 gets 0)
        names = [row[0] for row in user.venue_history[1:]]
        num_visits = np.array([row[1] for row in user.venue_history[1:]], dtype=float)

        # Identify venue to exclude
        idx_excluded = names.index("Venue_6")

        # Set its weight to zero
        num_visits_adj = num_visits.copy()
        num_visits_adj[idx_excluded] = 0

        # Normalize
        p_theoretical = num_visits_adj / num_visits_adj.sum()

        # Monte Carlo sampling
        N = 50000
        results = []

        for _ in range(N):
            x, y = user.get_return_position()

            # Identify venue name via coordinates
            for row in user.venue_history[1:]:
                if (row[2], row[3]) == (x, y):
                    results.append(row[0])
                    break

        # Empirical probabilities for ALL venues (including Venue_6)
        unique, counts = np.unique(results, return_counts=True)
        p_empirical = np.zeros(len(names))

        for i, name in enumerate(names):
            if name in unique:
                p_empirical[i] = counts[list(unique).index(name)] / N
            else:
                p_empirical[i] = 0.0  # Venue_6 should land here

        plt.figure(figsize=(8,5))
        plt.bar(names, p_empirical, alpha=0.7, label="Empirical")
        plt.plot(names, p_theoretical, "o-", label="Theoretical")
        plt.title("Distribution of return venues (including excluded one)")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        model.utils.filepath_save_plot("test_plots")

        np.testing.assert_allclose(
            p_empirical,
            p_theoretical,
            rtol=0.1,
            atol=0.01
        )


if __name__ == "__main__":
    unittest.main()
    





