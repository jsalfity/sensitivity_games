import numpy as np

from controller import LinearFeedbackController
from trajectory import Trajectory
from point_mass import PointMass
from quadratic import Quadratic

def main():
    # create dynamics
    dynamics = PointMass(m=1,
                    dt=0.1,
                    theta = {'m': 0})


    # create controller
    R = np.eye(dynamics.input_dim, dtype=int)
    Q = 3*np.eye(dynamics.state_dim, dtype=int)
    controller = LinearFeedbackController(dynamics, Q, R)

    #simulation conditions
    x0 = [1,0,1,0]
    T = 10

    # create and unroll trajectory
    traj = Trajectory(dynamics, T)
    X, U = traj.unroll(x0, controller)

    # compute costs
    # for x,u in zip(X,U)
        # evaluate cost at every relevant dimension


if __name__ == '__main__':
    main()