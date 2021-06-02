from dynamics import Dynamics

class Linear(Dynamics):
    def __init__(self, A, B, theta, K):
        super().__init__(state_dim=len(A),
                        input_dim=len(B[0]),
                        theta=theta)
        
        self.A_nominal = A
        self.B_nominal = B
        self.K = K

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
        return self.A_nominal
    
    def modify_B(self):
        '''
        Virtual
        '''
        return self.B_nominal
