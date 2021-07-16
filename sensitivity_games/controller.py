class Controller(object):
    '''
    '''
    def __init__(self,
                 dynamics):

        self.dynamics = dynamics

    def get_control(self, x, xf):
        '''
        Virtual function in abstract Controller class
        '''
        raise NotImplementedError('Not Implemented!')


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
