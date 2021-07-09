import argparse
import numpy as np

from sensitivity_games.linear_feedback_controller import (
    LinearFeedbackController)
from sensitivity_games.trajectory import Trajectory
from sensitivity_games.point_mass import PointMass
from sensitivity_games.quadratic import Quadratic
from sensitivity_games.trajectory_cost import TrajectoryCost

import torch.optim
from torch import Tensor

# converged K
converged_K = [1.0732,  0.9308, -0.1415,  0.4819, 
                0.7332,  1.0240,  0.1986,  0.3886]

# dare K
dare_K = [9.1704e-01,  1.6821e+00, -2.8130e-15, -1.4998e-15,
            -1.0866e-15, -1.3271e-15,  9.1704e-01,  1.6821e+00]


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--viz", default=True)
    parser.add_argument("--verbose", default=True)
    parser.add_argument("--T", default=20)
    parser.add_argument("-K", "--K", nargs='+', type=float, default=dare_K)
    parser.add_argument("--dm", default=0.0)

    return parser



# simulation conditions
# show_visualization = False
# print_stats = True
# epochs = 15000
x0 = Tensor([50, 0, 50, 0])
goal = [0, 0, 0, 0]
# T = 50
# learning_rate = 1e-3

def run_trajectory():

    parser = _setup_parser()
    args = parser.parse_args()

    # create dynamics
    dynamics = PointMass(m=1,
                         dt=0.1,
                         theta={'dm': Tensor([args.dm])})

    # create controller
    controller = LinearFeedbackController(dynamics)
    controller.K = Tensor(np.array(args.K).reshape(2,4))

    traj_cost = TrajectoryCost()
    traj_cost.addStateCost(Quadratic(n=goal[0], d=0))  # x position
    # traj_cost.addStateCost(Quadratic(n=goal[1], d=1))  # x velocity
    traj_cost.addStateCost(Quadratic(n=goal[2], d=2))  # y position
    # traj_cost.addStateCost(Quadratic(n=goal[3], d=3))  # y velocity

    traj_cost.addControlCost(Quadratic(n=0, d=0))  # x
    traj_cost.addControlCost(Quadratic(n=0, d=1))  # y

    # TODO: is this weight too high?
    traj_cost.addThetaCost('dm', Quadratic(n=0, d=0))

    # perturb with thetas
    dynamics.perturb()

    # unroll trajectory
    traj = Trajectory(dynamics, goal, args.T)
    X, U = traj.unroll(x0, controller)

    # do gradient update
    total_cost = traj_cost.evaluate(X, U, dynamics)

    # print info
    eigVals, _ = controller.get_eigenVals_eigenVecs()
    print("total_cost: {}".format(total_cost))
    print("K value: {}".format(controller.K))
    print("theta value: {}".format(dynamics.theta['dm']))
    print("eigVals: {}".format(eigVals))
    print(" ")
    print("________________________")

    if args.viz:
        traj.visualize()

if __name__ == "__main__":
    run_trajectory()
