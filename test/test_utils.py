import unittest
import numpy as np
import model.utils

class TestPowerLawSampling(unittest.TestCase):

    def test_power_law_sampling_shape(self):
        """Test if power_law_sampling returns array of correct shape"""
        n_samples = 1000
        samples = model.utils.power_law_sampling(par=0.55, min_val=1, max_val=100, n_samples=n_samples)
        self.assertEqual(samples.shape, (n_samples,))

    def test_power_law_sampling_values(self):
        """Test if all sampled values are within min_val and max_val"""
        n_samples = 1000
        min_val = 1
        max_val = 100
        samples = model.utils.power_law_sampling(par=0.55, min_val=min_val, max_val=max_val, n_samples=n_samples)
        self.assertTrue(np.all(samples >= min_val))
        self.assertTrue(np.all(samples <= max_val))

    def test_spatial_sampling_wrapper(self):
        """Test if spatial wrapper calls core and returns array"""
        samples = model.utils.spatial_power_law_sampling(par=0.55, min_val=5, max_val=100, n_samples=100)
        # Add check only if wrapper returns result
        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (100,))

    def test_temporal_sampling_wrapper(self):
        """Test if temporal wrapper calls core and returns array"""
        samples = model.utils.temporal_power_law_sampling(par=0.8, min_val=300, max_val=61200, n_samples=100)
        self.assertIsNotNone(samples)
        self.assertEqual(samples.shape, (100,))

    def test_radius_of_gyration(self):
        "test with dummies values if radius of gyration and center of mass are calculated correctly"
        x_traj = np.array([1,0,-1,0])
        y_traj = np.array([0,1,0,-1])

        r_cm, r_gyr = model.utils.radius_of_gyration(x_traj, y_traj)

        ##Da terminare!

        np.testing.assert_array_equal(r_cm, np.array([0, 0]))

    


if __name__ == "__main__":
    unittest.main()