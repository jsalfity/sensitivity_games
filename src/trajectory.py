import numpy as np

class Trajectory(object):
    '''
    '''
    def __init__(self, dynamics, T):
        
        self.x = []
        self.u = []

        self.dynamics = dynamics

        self.T = T
        self.nsteps = T / self.dynamics.dt

    def unroll(self, x0, controller):
        '''
        '''

        # initial conditions
        self.x.append(x0)
        self.u.append(np.zeros(controller.input_dim))

        for n in range(1, self.nsteps):

            self.x.append(self.dynamics.step(self.x[n-1], self.u[n-1]))
            self.u.append(controller.get_control(self.x[n]))

            if self.dynamics.done:
                break

        return self.x, self.u