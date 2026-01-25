import unittest
import model.user
import model.simulation
import pandas as pd
import pprint

class TestUser(unittest.TestCase):
    
    ##THIS SUPER METHOD IS SETTING THE SAME ISTANCE FOR ALL THE TESTS
    ##DO NOT MODIFY THE VALUES FOR THE CLASS. READ ONLY!
    @classmethod
    def setUpClass(cls):
        cls.simulation = model.simulation.Simulation(ro=0.6, gamma=-0.21, user_number= 1, time=5*17*3600)
        cls.simulation.run()

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

        original_venue_history = self.simulation.user.venue_history
        forced_venue_history = [["Venue", "Number of visits"], [1, 999], [2,1]]
        self.simulation.user.venue_history = forced_venue_history

        for i in range(10):
            
                event = self.simulation.user.get_return_venue()
                rare_event = 0

                if event == 2:
                    rare_event += 1
        self.assertLessEqual(rare_event, 1, msg=f"Test failed: Return velue selection is not working as expected."
                           f" Number of rare events is {rare_event}")
        
        self.simulation.user.venue_history = original_venue_history ##Restoring original value

    





