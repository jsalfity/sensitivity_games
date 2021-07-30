from sensitivity_games.linear import Linear
from matplotlib import pyplot as plt
import torch.tensor


class PointMass(Linear):
    def __init__(self, m, dt, theta):
        '''
        input:
        m (int): mass
        dt (double): time discrete
        theta (dict) = perturbation of nominal

        x1: x
        x2: x_dot
        x3: y
        x4: y_dot

        continuous time
        x' = A@x + B@u
        A = [[0 1 0 0 ],
             [0 0 0 0],
             [0 0 0 1],
             [0 0 0 0]]

        B = [[0 0],
             [1/m 0],
             [0 0],
             [0 1/m]]

        discrete time
        (x_(t+1)-x_t) / dt = A*x_t + B*u_t
        x_(t+1)-x_t = A*x_t*dt + B*u_t*dt
        x_(t+1) = (A*dt + eye) * x_t + B*u_t*dt

        A = [[1 dt 0 0],
             [0 1 0 0],
             [0 0 1 dt],
             [0 0 0 1]]

        B = [[0 0],
             [dt/m 0],
             [0 0],
             [0 dt/m]]
        '''
        self.dt = dt
        self.m = m
        self.theta = theta

        A = torch.tensor([[1, dt, 0, 0],
                         [0, 1.0, 0, 0],
                         [0, 0, 1.0, dt],
                         [0, 0, 0, 1.0]],
                         requires_grad=True)

        B = torch.tensor([[0, 0],
                         [dt/m, 0],
                         [0, 0],
                         [0, dt/m]],
                         requires_grad=True)
        # B = Tensor([[0, 0],
        #             [dt/m, 0],
        #             [0, 0],
        #             [0, dt/m]])

        super().__init__(A, B, theta)

    def step(self, x, u):
        '''
        '''
        return self.A@x + self.B@u

    def perturb(self):
        self.modify_A()
        self.modify_B()

    def done(self):
        '''
        '''
        return self.done

    def modify_A(self):
        '''
        '''
        # no modification of A
        return

    def modify_B(self):
        '''
        '''
        m = self.m * (1+self.theta['dm'])

        self.B = torch.tensor([[0, 0],
                              [self.dt/m, 0],
                              [0, 0],
                              [0, self.dt/m]],
                              requires_grad=True)

        # self.B = Tensor([[0, 0],
        #                 [self.dt/m, 0],
        #                 [0, 0],
        #                 [0, self.dt/m]])

    def visualize(self, X, U, xf, block):
        '''
        '''
        xmax = max(max([x[0].detach() for x in X]), xf[0])
        xmin = min(min([x[0].detach() for x in X]), xf[0])
        ymax = max(max([x[2].detach() for x in X]), xf[2])
        ymin = min(min([x[2].detach() for x in X]), xf[2])

        u1max = max([u[0].detach() for u in U])
        u1min = min([u[0].detach() for u in U])
        u2max = max([u[1].detach() for u in U])
        u2min = min([u[1].detach() for u in U])

        # plt.axis([xmin-2, xmax+2, ymin-2, ymax+2])
        # plt.grid()
        # plt.plot(goal[0], goal[2], 'xr')
        # for x, u in zip(X, U):
        #     plt.plot(x.detach()[0], x.detach()[2], '.b')
        #     plt.pause(0.001)
        # plt.close()

        fig, (x_ax, u_ax) = plt.subplots(2, 1)
        x_ax.grid()
        u_ax.grid()

        x_ax.axis([xmin-4, xmax+4, ymin-4, ymax+4])
        x_ax.plot(xf[0], xf[2], 'xr')
        x_ax.set_xlabel('x position')
        x_ax.set_ylabel('y position')

        u_ax.axis([0, len(U), min(u1min, u2min) - 4, max(u1max, u2max) + 4])
        u_ax.set_ylabel('control effort')
        u_ax.set_xlabel('Time step (dt)')
        t = 0
        for x, u in zip(X, U):
            x_ax.plot(x.detach()[0], x.detach()[2], '.b')
            u_ax.plot(t, u[0].detach(), '.r')
            u_ax.plot(t, u[1].detach(), '.k')
            u_ax.legend(['u0', 'u1'])
            t += 1
            plt.pause(0.001)

        plt.show(block=block)
        plt.close()
