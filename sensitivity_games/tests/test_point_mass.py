import unittest
import numpy as np
from unittest import TestCase
from sensitivity_games.experiments.point_mass_regulation import run_experiment


class Test_point_mass(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dynamics, cls.controller, cls.optimal_controller = run_experiment()

    def test_point_mass_experiment_K_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.controller.K.grad,
                                   np.zeros_like(self.controller.K.grad),
                                   atol=1e-07)
        return

    def test_point_mass_experiment_theta_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.dynamics.theta['dm'].grad,
                                   np.zeros_like(self.dynamics.theta['dm'].grad),
                                   atol=1e0-7)
        return

    def test_curvature_should_be_positive(cls):
        return

    def test_point_mass_should_converge_to_dare_k(self):
        np.allclose(self.controller.K.detach(),
                    self.optimal_controller.K.detach())
        return


if __name__ == '__main__':
    unittest.main()
