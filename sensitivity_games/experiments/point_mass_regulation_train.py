import argparse
import numpy as np
import random

from sensitivity_games.linear_feedback_controller import (
    LinearFeedbackController)
from sensitivity_games.trajectory import Trajectory
from sensitivity_games.point_mass import PointMass
from sensitivity_games.quadratic import Quadratic
from sensitivity_games.trajectory_cost import TrajectoryCost

import torch.optim
import torch.tensor


def _setup_parser():
    """Set up Python's ArgumentParser with params"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--viz", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--verbose_freq", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--x0_limit", type=float, default=5)
    parser.add_argument("--xf", nargs='+', type=float, default=[0, 0, 0, 0])
    parser.add_argument("--block", type=bool, default=False)

    return parser


def run_experiment():

    parser = _setup_parser()
    args = parser.parse_args()

    # initial conditions
    xf = torch.tensor(np.array(args.xf))

    # create dynamics
    dynamics = PointMass(m=1,
                         dt=0.1,
                         theta={'dm': torch.ones(1, requires_grad=True)})

    # create controller
    controller = LinearFeedbackController(dynamics, xf)

    traj_cost = TrajectoryCost()
    traj_cost.addStateCost(Quadratic(n=xf[0], d=0))  # x position
    # traj_cost.addStateCost(Quadratic(n=xf[1], d=1))  # x velocity
    traj_cost.addStateCost(Quadratic(n=xf[2], d=2))  # y position
    # traj_cost.addStateCost(Quadratic(n=xf[3], d=3))  # y velocity

    traj_cost.addControlCost(Quadratic(n=0, d=0))  # x
    traj_cost.addControlCost(Quadratic(n=0, d=1))  # y

    # TODO: is this weight too high?
    # traj_cost.addThetaCost('dm', Quadratic(n=0, d=0, weight=1e-5))

    optimizer_k = torch.optim.Adam([controller.K], lr=args.lr)
    optimizer_theta = torch.optim.Adam(dynamics.theta.values(),
                                       lr=args.lr)
    optimizer_theta.param_groups[0]['lr'] *= -1  # hack for gradient ascent

    for n in range(args.epochs):

        # perturb with thetas
        dynamics.perturb()

        # randomize x0
        x0 = torch.tensor([random.uniform(-args.x0_limit, args.x0_limit),
                           0,
                           random.uniform(-args.x0_limit, args.x0_limit),
                           0])

        # unroll trajectory
        traj = Trajectory(dynamics, xf, args.T)
        X, U = traj.unroll(x0, controller)

        # do gradient update
        total_cost = traj_cost.evaluate(X, U, dynamics)

        optimizer_k.zero_grad()
        optimizer_theta.zero_grad()

        # dtotal/dtheta[dm], dtotal/dstate_cost dtotal/dcontrol_cost
        total_cost.backward()

        optimizer_k.step()
        optimizer_theta.step()

        # print info
        if (n % args.verbose_freq == 0 and args.verbose) or n == args.epochs-1:
            # eigVals, _ = controller.get_eigenVals_eigenVecs()
            print("EPOCH: {}".format(n))
            print("total_cost: {}".format(total_cost))
            print("K value: {}".format(controller.K))
            print("K grad: {}".format(controller.K.grad))
            print("theta['dm'] value: {}".format(dynamics.theta['dm']))
            print("theta['dm'] grad: {}".format(dynamics.theta['dm'].grad))
            # print("eigVals: {}".format(eigVals))
            print("x0: {}".format(x0))
            print("xf (goal): {}".format(xf))
            print("xf (actual): {}".format(X[-1]))
            print(" ")
            print("________________________")

            if args.viz:
                traj.visualize(args.block)

    # check dare k
    R = np.eye(dynamics.input_dim, dtype=int)
    Q = np.eye(dynamics.state_dim, dtype=int)
    dare_k = controller.solve_dare_for_k(Q, R)

    if args.verbose:
        print("DARE K: {}".format(dare_k))

    # simulate an optimal trajectory
    optimal_controller = LinearFeedbackController(dynamics, xf)
    optimal_controller.K = dare_k
    optimal_eigVal, _ = optimal_controller.get_eigenVals_eigenVecs()
    optimal_traj = Trajectory(dynamics, xf, args.T)
    optimal_X, optimal_U = optimal_traj.unroll(x0, optimal_controller)
    optimal_total_cost = traj_cost.evaluate(optimal_X, optimal_U, dynamics)

    if args.verbose:
        print("DARE eigVals: {}".format(optimal_eigVal))
        print("DARE total_cost: {}".format(optimal_total_cost))

    if args.viz:
        optimal_traj.visualize(args.block)

    return dynamics, controller, optimal_controller, traj, traj_cost


if __name__ == "__main__":
    run_experiment()
