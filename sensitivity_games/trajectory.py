import torch

class Trajectory(object):
    '''
    '''
    def __init__(self, dynamics, xf, T):

        self.X = []
        self.U = []

        self.dynamics = dynamics

        self.T = T
        self.nsteps = int(T / self.dynamics.dt)
        self.xf = xf

    def unroll(self, x0, controller):
        '''
        '''
        # initial conditions
        self.X = [x0]
        self.U = [controller.get_control(x0)]

        for _ in range(1, self.nsteps):

            self.X.append(self.dynamics.step(self.X[-1], self.U[-1]))
            self.U.append(controller.get_control(self.X[-1]))

            # if self.dynamics.done(self.x[n-1]):
            #     break

        return self.X, self.U

    def visualize(self, block):
        '''
        '''
        self.dynamics.visualize(self.X, self.U, self.xf, block)
