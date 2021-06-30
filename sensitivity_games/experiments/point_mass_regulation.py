import numpy as np

from sensitivity_games.linear_feedback_controller import LinearFeedbackController
from sensitivity_games.trajectory import Trajectory
from sensitivity_games.point_mass import PointMass
from sensitivity_games.quadratic import Quadratic
from sensitivity_games.trajectory_cost import TrajectoryCost

import torch.optim
from torch import Tensor

# simulation conditions
show_visualization = False
print_stats = True
epochs = 15000
x0 = Tensor([5, 0, 5, 0])
goal = [0, 0, 0, 0]
T = 50
learning_rate = 1e-3


def run_experiment():
    # create dynamics
    dynamics = PointMass(m=1,
                         dt=0.1,
                         theta={'dm': torch.ones(1, requires_grad=True)})

    # create controller
    controller = LinearFeedbackController(dynamics)

    traj_cost = TrajectoryCost()
    traj_cost.addStateCost(Quadratic(n=goal[0], d=0))  # x position
    # traj_cost.addStateCost(Quadratic(n=goal[1], d=1))  # x velocity
    traj_cost.addStateCost(Quadratic(n=goal[2], d=2))  # y position
    # traj_cost.addStateCost(Quadratic(n=goal[3], d=3))  # y velocity

    traj_cost.addControlCost(Quadratic(n=0, d=0))  # x
    traj_cost.addControlCost(Quadratic(n=0, d=1))  # y

    traj_cost.addThetaCost('dm', Quadratic(n=0, d=0))

    optimizer_k = torch.optim.Adam([controller.K], lr=learning_rate)
    optimizer_theta = torch.optim.Adam(dynamics.theta.values(),
                                       lr=learning_rate)
    optimizer_theta.param_groups[0]['lr'] *= -1  # hack for gradient ascent

    for n in range(epochs):

        # perturb with thetas
        dynamics.perturb()

        # unroll trajectory
        traj = Trajectory(dynamics, goal, T)
        X, U = traj.unroll(x0, controller)

        # do gradient update
        total_cost = traj_cost.evaluate(X, U, dynamics)

        optimizer_k.zero_grad()
        optimizer_theta.zero_grad()

        total_cost.backward()

        optimizer_k.step()
        optimizer_theta.step()

        # print info
        if n % 100 == 0 and print_stats:
            eigVals, _ = controller.get_eigenVals_eigenVecs()
            print("EPOCH: {}".format(n))
            print("total_cost: {}".format(total_cost))
            print("K grad: {}".format(controller.K.grad))
            print("K value: {}".format(controller.K))
            print("theta grad: {}".format(dynamics.theta['dm'].grad))
            print("theta value: {}".format(dynamics.theta['dm']))
            print("dynamics B {}".format(dynamics.B))
            print("eigVals: {}".format(eigVals))
            print(" ")
            print("________________________")

            if show_visualization:
                traj.visualize()

    # check dare k
    R = np.eye(dynamics.input_dim, dtype=int)
    Q = np.eye(dynamics.state_dim, dtype=int)
    dare_k = controller.solve_dare_for_k(Q, R)

    if print_stats:
        print("dare K: {}".format(dare_k))

    # simulate an optimal trajectory
    optimal_controller = LinearFeedbackController(dynamics)
    optimal_controller.K = dare_k
    optimal_eigVal, _ = optimal_controller.get_eigenVals_eigenVecs()
    optimal_traj = Trajectory(dynamics, goal, T)
    optimal_X, optimal_U = optimal_traj.unroll(x0, optimal_controller)
    optimal_total_cost = traj_cost.evaluate(optimal_X, optimal_U, dynamics)

    if print_stats:
        print("eigVals: {}".format(optimal_eigVal))
        print("total_cost: {}".format(optimal_total_cost))

    if show_visualization:
        optimal_traj.visualize()

    return dynamics, controller, optimal_controller
