from linear import Linear
from torch import Tensor


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

        A = Tensor([[1, dt, 0, 0],
                    [0, 1.0, 0, 0],
                    [0, 0, 1.0, dt],
                    [0, 0, 0, 1.0]])

        B = Tensor([[0, 0],
                    [dt/m, 0],
                    [0, 0],
                    [0, dt/m]])

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

        self.B = Tensor([[0, 0],
                        [self.dt/m, 0],
                        [0, 0],
                        [0, self.dt/m]])
