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

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
