from controller import Controller
from scipy import linalg
import torch


class LinearFeedbackController(Controller):
    '''
    '''
    def __init__(self,
                 dynamics):

        super().__init__(dynamics)
        # self.K = torch.randn(dynamics.input_dim,
        #                      dynamics.state_dim,requires_grad=True)

        self.K = torch.tensor([[1.0, 0, 0, 0],
                              [0, 1.0, 0, 0]],
                              requires_grad=True)

    def solve_dare_for_k(self, Q, R):
        # http://www.mwm.im/lqr-controllers-with-python/
        # first, try to solve the ricatti equation
        X = torch.tensor(linalg.solve_discrete_are(self.dynamics.A,
                                                   self.dynamics.B,
                                                   Q,
                                                   R), dtype=torch.float32)
        # compute the LQR gain
        term1 = torch.tensor(linalg.inv(self.dynamics.B.T@X@self.dynamics.B+R),
                             dtype=torch.float32)

        term2 = self.dynamics.B.T@X@self.dynamics.A
        return term1@term2

    def get_eigenVals_eigenVecs(self):
        eigVals, eigVecs = linalg.eig(self.dynamics.A -
                                      self.dynamics.B@self.K.detach())
        return eigVals, eigVecs

    def get_control(self, x):
        '''
        '''
        return -self.K @ x
