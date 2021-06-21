from dynamics import Dynamics


class Linear(Dynamics):
    def __init__(self, A, B, theta):
        super().__init__(state_dim=len(A),
                         input_dim=len(B[0]),
                         theta=theta)

        self.A = A
        self.B = B

    def step(self, x, u):
        '''
        Discrete Time
        x'=A*x + B*u
        '''
        # redesign could be if slow
        # return self.modify_A() @ x + self.modify_B() @ u
        raise NotImplementedError('Not Implemented!')

    def modify_A(self):
        '''
        Virtual
        '''
        raise NotImplementedError('Not Implemented!')

    def modify_B(self):
        '''
        Virtual
        '''
        raise NotImplementedError('Not Implemented!')
