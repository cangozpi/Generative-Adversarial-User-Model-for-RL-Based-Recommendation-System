import torch
from torch import nn

# This model takes in the old state and the newly chosen action as input and produces the new state representation
# using LSTM(Long Short Term Memory)
class History_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        input_size (int): feature_dim of the actions.
        hidden_size (int): dimension of the state representation vector (dim of output)
        num_layers (int): number of recurrent layers in the LSTM model.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_layers = num_layers
        self.state_dim = hidden_size
        self.lstm_model = torch.nn.LSTM(input_size, self.state_dim, self.num_layers, batch_first=True).to(self.device)


    def forward(self, actions):
        """
        Inputs:
            new_action (torch.Tensor): action chosen by the user (either ground truth action or Generator_UserModel generated action).
            [batch_size (#users), num_time_steps, feature_dim]
        Returns:
            new_state (torch.Tensor): old_state updated after taking new_action (i.e. updated history representation). 
            [batch_size (#users), num_time_steps, state_dim]
            (h, c) (tuple): hidden and cell states. 
            Note that the returned new_state tensor is of same shape as the old_state tensor. 
        """
        out, _ = self.lstm_model(actions)
        return out 
