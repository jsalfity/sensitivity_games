from sensitivity_games.controller import Controller


class RegulatorController(Controller):
    '''
    '''
    def __init__(self,
                 dynamics,
                 xf):
        super().__init__(dynamics)
        self.xf = xf
