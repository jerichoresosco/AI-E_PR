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
        """Test if the type of user_history after .run is pd.DataFrame"""
        self.assertIsInstance(self.simulation.user.user_history, pd.DataFrame)

    def test_venue_history_type(self):
        """Test if the type of venue_history after .run is pd.DataFrame"""
        self.assertIsInstance(self.simulation.user.venue_history, pd.DataFrame)

    def test_trajectory_continuity(self):
        """Test if there is continuity between positions of user"""
        user_history = self.simulation.user.user_history
        
        for i in range(len(user_history) - 2):
            end_pos = user_history.iloc[i+1, 2]     
            start_pos = user_history.iloc[i+2, 1]    
            self.assertEqual(end_pos, start_pos)

    def test_get_return_venue(self):
        """Test with a correctness of 99.996% if return venue selects the most probable venue"""

        forced_venue_history = [["Venue", "Number of visits"], [1, 999], [2,1]]
        self.simulation.user.venue_history = forced_venue_history

        rare_event = 0
        for i in range(10):
            
                event = self.simulation.user.get_return_venue()
                
                if event == 2:
                    rare_event += 1
        self.assertLessEqual(rare_event, 1, msg=f"Test failed: Return velue selection is not working as expected."
                           f" Number of rare events is {rare_event}")
        


    def test_get_return_venue(self):
        """
        Statistichal test to check if get_return_venue is working properly. The test will
        be conducted using a hand-written venue_history, checking if actual return values
        match the theorical prediction. In this test will be choosen a current position which
        is not inside venue history.
        """

        venue_history = [
            ["Venue", "Number of visits"],
            (1, 5),
            (2, 4),
            (3, 3),
            (4, 2),
            (5, 1),
        ]

        self.simulation.user.venue_history = venue_history
        self.simulation.user.position = 999 #Choosing a current position which will not interfere

        # theoretical probability
        num_visits = np.array([v[1] for v in venue_history[1:]], dtype=float)
        p_theoretical = num_visits / num_visits.sum()

        N = 50_000 #Number of extractions
        results = []

        for _ in range(N):
            r = self.simulation.user.get_return_venue()
            results.append(r)  

        results = np.array(results)

        values, counts = np.unique(results, return_counts=True)
        p_empirical = counts / N

        plt.figure(figsize=(8, 5))
        plt.bar(values, p_empirical, alpha=0.7, label='Empirical frequency')
        plt.plot(range(1,len(venue_history)), p_theoretical, 'o-', label='Theoretical probability', markersize=8)
        plt.xlabel("Venue index in venue_history")
        plt.ylabel("Probability")
        plt.title("Return venue distribution: empirical vs theoretical")
        plt.legend()
        plt.tight_layout()
        model.utils.filepath_save_plot("test_plots")

        np.testing.assert_allclose(p_empirical, p_theoretical, rtol=0.1, atol=0.01)


if __name__ == "__main__":
    unittest.main()
    





