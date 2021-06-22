class Trajectory(object):
    '''
    '''
    def __init__(self, dynamics, goal, T):

        self.X = []
        self.U = []

        self.dynamics = dynamics

        self.T = T
        self.nsteps = int(T / self.dynamics.dt)
        self.goal = goal

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

    def visualize(self):
        '''
        '''
        self.dynamics.visualize(self.X, self.U, self.goal)
        # xmax=max([x[0].detach() for x in self.X])
        # xmin=min([x[0].detach() for x in self.X])
        # ymax=max([x[2].detach() for x in self.X])
        # ymin=min([x[2].detach() for x in self.X])

        # plt.axis([xmin-2, xmax+2, ymin-2, ymax+2])
        # plt.grid()
        # plt.plot(self.goal[0],self.goal[2],'xr')
        # for x, u in zip(self.X, self.U):
        #     plt.plot(x.detach()[0], x.detach()[2],'.b')
        #     plt.pause(0.001)
        # plt.close()
