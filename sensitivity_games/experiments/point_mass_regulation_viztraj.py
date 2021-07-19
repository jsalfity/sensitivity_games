import argparse
import numpy as np
import random

from sensitivity_games.linear_feedback_controller import (
    LinearFeedbackController)
from sensitivity_games.trajectory import Trajectory
from sensitivity_games.point_mass import PointMass
from sensitivity_games.quadratic import Quadratic
from sensitivity_games.trajectory_cost import TrajectoryCost

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

    parser.add_argument("--viz", type=bool, default=True)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--K", nargs='+', type=float, default=dare_K)
    parser.add_argument("--dm", type=float, default=0.0)
    parser.add_argument("--x0", nargs='+', type=float, default=[5, 0, 5, 0])
    parser.add_argument("--xf", nargs='+', type=float, default=[0, 0, 0, 0])
    parser.add_argument("--block", type=bool, default=True)

    return parser


def run_trajectory():

    parser = _setup_parser()
    args = parser.parse_args()

    # initial conditions
    # x0 = Tensor(np.array(args.x0))
    x0 = Tensor([random.random(-5, 5), 0, random.random(-5, 5), 0])
    xf = Tensor(np.array(args.xf))

    # create dynamics
    dynamics = PointMass(m=1,
                         dt=0.1,
                         theta={'dm': Tensor([args.dm])})

    # create controller
    controller = LinearFeedbackController(dynamics, xf)
    controller.K = Tensor(np.array(args.K).reshape(2, 4))

    traj_cost = TrajectoryCost()
    traj_cost.addStateCost(Quadratic(n=xf[0], d=0))  # x position
    # traj_cost.addStateCost(Quadratic(n=xf[1], d=1))  # x velocity
    traj_cost.addStateCost(Quadratic(n=xf[2], d=2))  # y position
    # traj_cost.addStateCost(Quadratic(n=xf[3], d=3))  # y velocity

    traj_cost.addControlCost(Quadratic(n=0, d=0))  # x
    traj_cost.addControlCost(Quadratic(n=0, d=1))  # y

    # TODO: is this weight too high?
    traj_cost.addThetaCost('dm', Quadratic(n=0, d=0))

    # perturb with thetas
    dynamics.perturb()

    # unroll trajectory
    traj = Trajectory(dynamics, xf, args.T)
    X, U = traj.unroll(x0, controller)

    # evaluate cost
    total_cost = traj_cost.evaluate(X, U, dynamics)

    # print info
    eigVals, _ = controller.get_eigenVals_eigenVecs()
    print("total_cost: {}".format(total_cost))
    print("K value: {}".format(controller.K))
    print("theta value: {}".format(dynamics.theta['dm']))
    print("eigVals: {}".format(eigVals))
    print("x0: {}".format(x0))
    print("xf (goal): {}".format(xf))
    print("xf (actual): {}".format(X[-1]))
    print("T: {}".format(args.T))
    print(" ")
    print("________________________")

    if args.viz:
        traj.visualize(args.block)


if __name__ == "__main__":
    run_trajectory()
