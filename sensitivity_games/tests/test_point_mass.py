import unittest
from unittest import TestCase

from torch import Tensor
import numpy as np
import random

from sensitivity_games.experiments.point_mass_regulation import run_experiment


class Test_Point_Mass(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dynamics, cls.controller, cls.optimal_controller, \
            cls.traj, cls.traj_cost = run_experiment()

    def test_point_mass_experiment_K_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.controller.K.grad,
                                   np.zeros_like(self.controller.K.grad),
                                   atol=1e-07)
        return

    def test_point_mass_experiment_theta_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.dynamics.theta['dm'].grad,
                                   np.zeros_like(
                                       self.dynamics.theta['dm'].grad),
                                   atol=1e-07)
        return

    def test_controller_k_curvature_should_be_positive(self):
        # K dimension: 2x4
        x0 = Tensor([5, 0, 5, 0])

        X, U = self.traj.unroll(x0, self.controller)
        converged_K = self.controller.K.detach()
        converged_K_cost = self.traj_cost.evaluate(X, U, self.dynamics)

        for _ in range(0, 100):
            self.controller.K = _random_perturb(converged_K.clone())
            X, U = self.traj.unroll(x0, self.controller)
            perturbed_K_cost = self.traj_cost.evaluate(X, U, self.dynamics)
            if perturbed_K_cost < converged_K_cost:
                return False

        return True

    def test_point_mass_should_converge_to_dare_k(self):
        np.allclose(self.controller.K.detach(),
                    self.optimal_controller.K.detach())
        return


def _random_perturb(K, bound=1):
    '''
    '''
    for irow, row in enumerate(K):
        for icol, col in enumerate(row):
            K[irow][icol] = K[irow][icol]+random.uniform(-bound, bound)

    return K


if __name__ == '__main__':
    unittest.main()
