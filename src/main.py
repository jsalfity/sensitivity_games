import numpy as np

from linear_feedback_controller import LinearFeedbackController
from trajectory import Trajectory
from point_mass import PointMass
from quadratic import Quadratic
from trajectory_cost import TrajectoryCost

import torch.optim
from torch import Tensor


def main():
    # create dynamics
    dynamics = PointMass(m=1,
                         dt=0.1,
                         theta={'dm': torch.ones(1, requires_grad=True)})

    # create controller
    controller = LinearFeedbackController(dynamics)

    # simulation conditions
    epochs = 5000
    x0 = Tensor([1, 0, 1, 0])
    T = 10

    traj_cost = TrajectoryCost()
    traj_cost.addStateCost(Quadratic(n=0, d=0))  # x position
    # traj_cost.addStateCost(Quadratic(n=0,d=1))  # x velocity
    # traj_cost.addStateCost(Quadratic(n=0,d=2))  # y position
    # traj_cost.addStateCost(Quadratic(n=0,d=3))  # y velocity

    traj_cost.addControlCost(Quadratic(n=0, d=0))  # x
    traj_cost.addControlCost(Quadratic(n=0, d=1))  # y

    traj_cost.addThetaCost('dm', Quadratic(n=0, d=0))

    learning_rate = 1e-3
    optimizer_k = torch.optim.Adam([controller.K], lr=learning_rate)
    optimizer_theta = torch.optim.Adam(dynamics.theta.values(),
                                       lr=learning_rate)
    optimizer_theta.param_groups[0]['lr'] *= -1  # hack for gradient ascent

    for n in range(epochs):

        # perturb with thetas
        dynamics.perturb()

        # unroll trajectory
        traj = Trajectory(dynamics, T)
        X, U = traj.unroll(x0, controller)

        # do gradient update
        total_cost = traj_cost.evaluate(X, U, dynamics)

        optimizer_k.zero_grad()
        optimizer_theta.zero_grad()

        total_cost.backward()

        optimizer_k.step()
        optimizer_theta.step()

        # print info
        if n % 100 == 0:
            print("epoch: {}, total_cost: {}".format(n, total_cost))
            print("K grad: {}".format(controller.K.grad))
            print("K value: {}".format(controller.K))
            print("theta grad: {}".format(dynamics.theta['dm'].grad))
            print("theta value: {}".format(dynamics.theta['dm']))
            print("dynamics B {}".format(dynamics.B))
            print(" ")

    # check dare k
    R = np.eye(dynamics.input_dim, dtype=int)
    Q = np.eye(dynamics.state_dim, dtype=int)
    dare_k = controller.solve_dare_for_k(Q, R)
    print("dare K: {}".format(dare_k))


if __name__ == '__main__':
    main()
