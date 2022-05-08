import torch
from torch import nn

# This model takes in the old state and the newly chosen action as input and produces the new state representation
# using LSTM(Long Short Term Memory)
class History_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, old_state, new_action):
        """
        Inputs:
            old_state (torch.Tensor): previous output of the historyLSTM. Vector Representation of the history
            new_action (torch.Tensor): action chosen by the user (either ground truth action or Generator_UserModel generated action)
        Returns:
            new_state (torch.Tensor): old_state updated after taking new_action (i.e. updated history representation). 
            Note that the returned new_state tensor is of same shape as the old_state tensor.
        """
        pass