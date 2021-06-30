
class TrajectoryCost(object):
    def __init__(self):
        self.state_costs = []
        self.control_costs = []
        self.theta_costs = {}  # key: 'dm', value: list(Quadratic)

    def addStateCost(self, cost_function):
        '''
        '''
        self.state_costs.append(cost_function)

    def addControlCost(self, cost_function):
        '''
        '''
        self.control_costs.append(cost_function)

    def addThetaCost(self, key, cost_function):
        '''
        '''
        if key not in self.theta_costs.keys():
            self.theta_costs[key] = []

        self.theta_costs[key].append(cost_function)

    def evaluate(self, X, U, dynamics):
        '''
        '''
        total = 0.0

        # compute costs
        for x, u in zip(X, U):

            total += sum(c.evaluate(x) for c in self.state_costs)
            total += sum(c.evaluate(u) for c in self.control_costs)
            total -= sum(sum(c.evaluate(v) for c in self.theta_costs[k])
                         for k, v in dynamics.theta.items())

        return total
