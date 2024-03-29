from sensitivity_games.regulator_controller import RegulatorController
from scipy import linalg
import torch


class LinearFeedbackController(RegulatorController):
    '''
    '''
    def __init__(self,
                 dynamics,
                 xf):
        super().__init__(dynamics, xf)

        self.K = torch.tensor([[1.0, 0, 0, 0],
                              [0, 1.0, 0, 0]],
                              requires_grad=True)

    def solve_dare_for_k(self, Q, R):
        '''
        Calculation taken from http://www.mwm.im/lqr-controllers-with-python/
        '''
        X = torch.tensor(linalg.solve_discrete_are(self.dynamics.A,
                                                   self.dynamics.B.detach(),
                                                   Q,
                                                   R), dtype=torch.float32)

        # compute the LQR gain
        term1 = torch.tensor(
            linalg.inv((self.dynamics.B.T @ X @ self.dynamics.B).detach() + R),
            dtype=torch.float32)

        term2 = self.dynamics.B.T.detach() @ X @ self.dynamics.A

        K = term1 @ term2
        return K

    def get_eigenVals_eigenVecs(self):
        '''
        '''
        eigVals, eigVecs = linalg.eig(self.dynamics.A -
                                      self.dynamics.B.detach()@self.K.detach())
        return eigVals, eigVecs

    def get_control(self, x):
        '''
        '''
        return -self.K @ (x - self.xf)
