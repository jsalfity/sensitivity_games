from cost import Cost

class Quadratic(Cost):
    def __init__(self, weight, n, d):
        '''
        Quadratic function acts on a single state

        input:
        weight (int): quadratic cost value
        n (double): nominal value
        d (int): dimension of z to act on
        '''
        super().__init__(weight)

        self.n = n
        self.d = d

    def evaluate(self, z):
        '''
        input:
        z (int): input dimension

        return:
        quadratic cost
        '''
        return self.weight * (self.n-z[self.d])**2