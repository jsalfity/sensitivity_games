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
