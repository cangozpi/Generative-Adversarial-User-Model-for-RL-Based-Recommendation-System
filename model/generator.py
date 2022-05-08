import torch
from torch import nn

# Note that User Model is the Generator in this context
class Generator_UserModel(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, state, displayed_items):
        """
        Input:
            Takes in state (vector representation of history), and 
            displayed_items (Tensor consisting of the feature vector embeddings) which are the possible items to display
            at the current time step.
        Return:
            Action that the user would take (i.e. item clicked(or nothing clicked at all) by the user)
        """
        pass