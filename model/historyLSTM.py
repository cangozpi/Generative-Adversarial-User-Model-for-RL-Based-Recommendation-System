import torch
from torch import nn

# This model takes in the old state and the newly chosen action as input and produces the new state representation
# using LSTM(Long Short Term Memory)
class History_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.num_layers = num_layers
        self.state_dim = hidden_size
        self.lstm_model = torch.nn.LSTM(input_size, self.state_dim, num_layers, batch_first=True).to(self.device)
        
        # ınitialize hidden and cell state
        self.h0 = torch.zeros(self.num_layers, input_size.shape[0], self.state_dim).to(self.device)
        self.c0= torch.zeros(self.num_layers, input_size.shape[0], self.state_dim).to(self.device)


    def forward(self, actions):
        """
        Inputs:
            new_action (torch.Tensor): action chosen by the user (either ground truth action or Generator_UserModel generated action).
            [batch_size (#users), num_time_steps, feature_dim]
        Returns:
            new_state (torch.Tensor): old_state updated after taking new_action (i.e. updated history representation). 
            [batch_size (#users), num_time_steps, state_dim] 
            Note that the returned new_state tensor is of same shape as the old_state tensor. 
        """
        out, _ = self.lstm_model(actions, (self.h0, self.c0))
        return out 
