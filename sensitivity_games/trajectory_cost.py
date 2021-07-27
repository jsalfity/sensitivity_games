# import torch

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

        # compute theta costs
        total -= sum(sum(c.evaluate(v) for c in self.theta_costs[k])
                     for k, v in dynamics.theta.items())

        # compute trajectory costs
        for x, u in zip(X, U):
            total += sum(c.evaluate(x) for c in self.state_costs)
            total += sum(c.evaluate(u) for c in self.control_costs)

        return total

    def evaluate_grad2(self):
        '''
        '''
        # if you can't get this, move on

        # https://github.com/HJReachability/ilqgames/blob/72b0e4f1803449bc00cfe7d3903920eb9fd376a6/python/player_cost.py#L108

        # https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-network/15270/21

        # set up total cost function as torch variable
        # take grad wrt state and control, careful of dimension
        # grad_state = torch.autograd.grad(self.total, self.state)
        # grad2_state = torch.autograd.grad(grad_state, self.state)

        # grad_control = torch.autograd.grad(self.total, self.control,
        #                                    create_graph = True)
        # grad2_control = torch.autograd.grad(grad_control, self.control,
        #                                     create_graph = True)

        # https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
        # torch.autograd.functional.hessian(func, inputs, create_graph=False,
        #                                   strict=False, vectorize=False)

        # return grad2_state, grad2_control
