import unittest
from model.user import User
from model.simulation import Simulation

class TestUser(unittest.TestCase):
    
    ##THIS SUPER METHOD IS SETTING THE SAME ISTANCE FOR ALL THE TESTS
    ##DO NOT MODIFY THE VALUES FOR THE CLASS. READ ONLY
    @classmethod
    def setUpClass(cls):
        cls.simulation = Simulation(ro=0.6, gamma=-0.21, time=5*17*3600)
        cls.simulation.run()

    def test_user_history_type(self):
        """Test if the type of user_history is list"""
        self.assertIsInstance(self.simulation.user.user_history, list)

    def test_venue_history_type(self):
        """Test if the type of venue_history is list"""
        self.assertIsInstance(self.simulation.user.venue_history, list)

    def test_trajectory_continuity(self):
        """Test if there is continuity between positions of user"""
        for i in range(len(self.simulation.user.user_history) - 2):
            self.assertEqual(self.simulation.user.user_history[i+1][2], self.simulation.user.user_history[i+2][1])

    ###THIS TEST MUST BE THE LAST ONE OF THE SERIES OF TESTUSER, BECAUSE HERE THE VALUE OF
    ###OF VENUE_HISTORY WILL BE MANUALLY CHANGED AND EVERY TEST AFTER THIS WILL SHARE THE SAME CHANGE.
    def test_get_return_venue(self):
        """Test with a correctness of 99.996% if return venue selects the most probable venue"""

        forced_venue_history = [["Venue", "Number of visits"], [1, 999], [2,1]]
        self.simulation.user.venue_history = forced_venue_history

        for i in range(10):
            
                event = self.simulation.user.get_return_venue()
                rare_event = 0

                if event == 2:
                    rare_event += 1
        self.assertLessEqual(rare_event, 1, msg=f"Test failed: Return velue selection is not working as expected."
                           f" Number of rare events is {rare_event}")
                


