from sensitivity_games.controller import Controller


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
