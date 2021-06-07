class Controller(object):
    '''
    '''
    def __init__(self, 
                dynamics):
        self.dynamics = dynamics

    def get_control(self, x):
        '''
        Virtual function in abstract Controller class
        '''
        raise NotImplementedError('Not Implemented!')

from scipy import linalg
import numpy as np
class LinearFeedbackController(Controller):
    '''
    '''
    def __init__(self,
                dynamics,
                Q,
                R):
        super().__init__(dynamics)
        self.Q = Q
        self.R = R

        # http://www.mwm.im/lqr-controllers-with-python/
        # first, try to solve the ricatti equation
        X = np.matrix(linalg.solve_discrete_are(self.dynamics.A, 
                                                self.dynamics.B, 
                                                self.Q, 
                                                self.R))
        # compute the LQR gain
        self.K = np.matrix(linalg.inv(self.dynamics.B.T*X*self.dynamics.B+R)*(self.dynamics.B.T*X*self.dynamics.A))

    def get_control(self, x):
        '''
        '''
        return -self.K @ x

class NeuralNetworkController(Controller):
    '''
    '''
    def __init__(self,
                dynamics,
                model):
        super().__init__(dynamics)        
        self.model = model
    
    def get_control(self, x):
        '''
        '''
        return self.model.forward(x)