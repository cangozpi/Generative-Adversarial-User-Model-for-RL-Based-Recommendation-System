import torch
from torch import nn

# Note that Reward Generating model is the Discriminator in this context
class Discriminator_RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            #TODO: fill the model
        ]
        self.model = torch.nn.Sequential(*layers)
        pass

    def forward(self, state, action):
        """
        Inputs:
            state (torch.Tensor): Vector representing the current state (history). This tensor is the output 
                of the historyLSMT model.
            action (torch.Tensor): Action chosen by the user (either ground truth action, or Generator_UserModel generated action).
                Feature vector representation of the chosen action (clicked item).
        Returns:
            reward (torch.float): reward value for taking the action at the given state.
        """
        pass