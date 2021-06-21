
class Dynamics(object):
    '''
    '''
    def __init__(self,
                 state_dim,
                 input_dim,
                 theta):

        # members
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.theta = theta

    def step(self, x, u):
        '''
        Virtual function in abstract Dynamics class

        input: x (np.array) state
        input: u (np.array) action
        '''
        raise NotImplementedError('Not Implemented!')

    def perturb(self):
        '''
        Virtual function in abstract Dynamics class
        '''
        raise NotImplementedError('Not Implemented!')

    def done(self, x):
        '''
        Virtual function in abstract Dynamics class
        '''
        return False
