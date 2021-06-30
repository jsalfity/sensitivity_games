from sensitivity_games.cost import Cost


class Quadratic(Cost):
    def __init__(self, n, d, weight=1):
        '''
        Quadratic function acts on a single state

        input:
        weight (double): quadratic cost value
        n (double): nominal value
        d (int): dimension of z to act on
        '''
        super().__init__(weight)

        self.n = n
        self.d = d

    def evaluate(self, z):
        '''
        input:
        z (list): input to cost

        return:
        quadratic cost
        '''
        return self.weight * (self.n-z[self.d])**2
