import unittest
import numpy as np
from model.utils import power_law_sampling, spatial_power_law_sampling, temporal_power_law_sampling
from model.user import User
from model.simulation import Simulation

class TestPowerLawSampling(unittest.TestCase):

    def test_power_law_sampling_shape(self):
        """Test if power_law_sampling returns array of correct shape"""
        n_samples = 1000
        samples = power_law_sampling(par=0.55, min_val=1, max_val=100, n_samples=n_samples)
        self.assertEqual(samples.shape, (n_samples,))

    def test_power_law_sampling_values(self):
        """Test if all sampled values are within min_val and max_val"""
        n_samples = 1000
        min_val = 1
        max_val = 100
        samples = power_law_sampling(par=0.55, min_val=min_val, max_val=max_val, n_samples=n_samples)
        self.assertTrue(np.all(samples >= min_val))
        self.assertTrue(np.all(samples <= max_val))

    def test_spatial_sampling_wrapper(self):
        """Test if spatial wrapper calls core and returns array"""
        samples = spatial_power_law_sampling(par=0.55, min_val=5, max_val=100, n_samples=100)
        # Add check only if wrapper returns result
        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (100,))

    def test_temporal_sampling_wrapper(self):
        """Test if temporal wrapper calls core and returns array"""
        samples = temporal_power_law_sampling(par=0.8, min_val=300, max_val=61200, n_samples=100)
        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (100,))

class TestUser(unittest.TestCase):
    
    ##THIS SUPER METHOD IS SETTING THE SAME ISTANCE FOR ALL THE TESTS
    ##DO NOT MODIFY THE VALUES FOR THE CLASS. READ ONLY
    @classmethod
    def setUpClass(cls):
        cls.simulation = Simulation(ro=0.6, gamma=-0.21, time=5*17*3600)
        cls.simulation.run()

    def test_user_history_type(self):
        self.assertIsInstance(self.simulation.user.user_history, list)

    def test_venue_history_type(self):
        self.assertIsInstance(self.simulation.user.venue_history, list)

    def test_trajectory_continuity(self):
        for i in range(len(self.simulation.user.user_history) - 2):
            self.assertEqual(self.simulation.user.user_history[i+1][2], self.simulation.user.user_history[i+2][1])


if __name__ == "__main__":
    unittest.main()