class Trajectory(object):
    '''
    Compute costs
    '''
    def __init__(self, dynamics, x0, u0, T):

        self.x0 = x0
        self.u0 = u0

        self.x = x0
        self.u = u0

        self.dynamics = dynamics
        self.state_costs = []
        self.control_costs = []

        self.t = 0
        self.T = T


    def unroll(self):        
        '''
        Perturb, 
        unroll trajectory, 
        compute cost,
        compute gradients
        '''

        self.dynamics.perturb() # should be done by player 2?

        while self.t < self.T and not self.dynamics.done :
            
            # compute state costs
            
            # control costs

            # compute gradients, 
            # get saved for player 2?

            self.x = self.dynamics.step(self.x, self.u)
            self.u = self.dynamics.get_control(self.x)
            
            self.t += self.dynamics.dt

        return sum(self.state_costs)+sum(self.control_costs)