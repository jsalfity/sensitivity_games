import unittest
from unittest import TestCase

import torch
from torch import Tensor
import numpy as np
import random

from sensitivity_games.experiments.point_mass_regulation_train import (
    run_experiment)
from sensitivity_games.linear_feedback_controller import (
    LinearFeedbackController)
from sensitivity_games.point_mass import PointMass

zero_tolerance = 1e-02
perturb_bound = 0.5
n_perturbations = 1000


class Test_Point_Mass(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dynamics, cls.controller, cls.optimal_controller, \
            cls.traj, cls.traj_cost = run_experiment()

    def test_point_mass_experiment_K_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.controller.K.grad,
                                   np.zeros_like(self.controller.K.grad,
                                                 dtype=float),
                                   atol=zero_tolerance)
        return

    def test_point_mass_experiment_theta_grad_should_converge_to_zero(self):
        np.testing.assert_allclose(self.dynamics.theta['dm'].grad,
                                   np.zeros_like(
                                       self.dynamics.theta['dm'].grad,
                                       dtype=float),
                                   atol=zero_tolerance)
        return

    def test_controller_k_curvature_should_be_positive(self):
        # K dimension: 2x4
        x0 = Tensor([5, 0, 5, 0])

        X, U = self.traj.unroll(x0, self.controller)
        converged_K_cost = self.traj_cost.evaluate(X, U, self.dynamics)

        # make a copy to not alter self.controller.K
        test_controller = LinearFeedbackController(self.dynamics)

        for _ in range(0, n_perturbations):
            # set equal to converged self.controller.K
            test_controller.K = self.controller.K.clone()

            # perturb
            test_controller.K = _random_perturb_K(test_controller.K.detach())

            # unroll and calculate cost
            X, U = self.traj.unroll(x0, test_controller)
            perturbed_K_cost = self.traj_cost.evaluate(X, U, self.dynamics)

            if perturbed_K_cost < converged_K_cost:
                return False

        return True

    def test_dm_curvature_should_be_positive(self):

        x0 = Tensor([5, 0, 5, 0])

        X, U = self.traj.unroll(x0, self.controller)
        converged_dm_cost = self.traj_cost.evaluate(X, U, self.dynamics)

        # make a copy to not alter self.dynamics
        test_dynamics = PointMass(m=1,
                                  dt=0.1,
                                  theta={'dm': torch.ones(1,
                                                          requires_grad=True)})

        for _ in range(0, n_perturbations):
            # set equal to converged self.dynamics.theta['dm']
            test_dynamics.theta['dm'] = self.dynamics.theta['dm'].clone()

            # perturb
            test_dynamics.theta['dm'] = _random_perturb_dm(
                                            test_dynamics.theta['dm'].detach()
                                            )

            X, U = self.traj.unroll(x0, self.controller)
            perturbed_dm_cost = self.traj_cost.evaluate(X, U, test_dynamics)

            if perturbed_dm_cost < converged_dm_cost:
                return False

        return True

    def test_point_mass_should_converge_to_dare_k(self):
        # TODO: push out T until close, find 'infinite' T
        np.testing.assert_allclose(self.controller.K.detach(),
                                   self.optimal_controller.K.detach(),
                                   atol=zero_tolerance)
        return


def _random_perturb_K(K, bound=perturb_bound):
    # Note: due to local optimization,
    # we want dense sample of K (4x2) perturbations

    for irow, row in enumerate(K):
        for icol, col in enumerate(row):
            K[irow][icol] = K[irow][icol]+random.uniform(-bound, bound)

    return K


def _random_perturb_dm(dm, bound=perturb_bound):
    return dm+random.uniform(-bound, bound)


if __name__ == '__main__':
    unittest.main()
