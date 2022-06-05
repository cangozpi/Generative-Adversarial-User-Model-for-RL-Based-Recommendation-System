import torch
from torch import nn

# Note that Reward Generating model is the Discriminator in this context
class Discriminator_RewardModel(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, hidden_dim):
        """
        input_size: should equal (num_displayed_items*feature_dims) + state_dim.
        output_size: should equal (num_displayed_items+1). 
        n_hidden: number of hidden layers of the Discriminator model's MLP.
        hidden_dim: hidden dimension of the layers of the Discriminator model's MLP.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        layers = []

        layers.extend([torch.nn.Linear(input_size, hidden_dim),torch.nn.ReLU()])
        
        for n in range(n_hidden-1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim),torch.nn.ReLU()])
            
            
        # Regression Layer (outputs score for each display_item)
        # Note that output_dim equals the number of possible actions
        layers.extend([torch.nn.Linear(hidden_dim, output_size), torch.nn.Tanh()])
         
        self.model = torch.nn.Sequential(*layers) # (inp_0 inp_1 .. inp_k) --> classification(inp_0, inp_1, .. inp_k) 

    def forward(self, state, displayed_items):
        """
        Inputs:
            Input:
                state (torch.Tensor): [max(num_time_steps), state_dim]
                displayed_items (torch.Tensor): [max(num_time_steps), num_displayed_item, feature_dim]
            Returns:
                reward (torch.float): reward value for taking the action at the given state. 
                [batch_size (#users), num_time_steps, (num_displayed_items+1)]
        """
        # Convert rnn.PackedSequences to simple Tensors
        if isinstance(state, torch.nn.utils.rnn.PackedSequence):
            state, _ = torch.nn.utils.rnn.pad_packed_sequence(state, batch_first=True)
        if isinstance(displayed_items, torch.nn.utils.rnn.PackedSequence):
            displayed_items, lens_displayed_item = torch.nn.utils.rnn.pad_packed_sequence(displayed_items, batch_first=True)
        
        # Prepare input
        batch_size = state.shape[0] # B
        num_time_steps = displayed_items.shape[1] # L
        # concat zero vector to displayed items to represent user not clicking on any of the displayed items
        not_clicking_feature_vec = torch.zeros((batch_size, num_time_steps, 1, displayed_items.shape[-1])) # --> [batch_size (#users), max(num_time_steps), 1, feature_dim]
        displayed_items = torch.cat((displayed_items.to(self.device), not_clicking_feature_vec.to(self.device)), -2) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1), feature_dims]
        displayed_items_flat = displayed_items.view(batch_size, num_time_steps, -1) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)*feature_dims]
        input_features = torch.cat((displayed_items_flat, state), dim=-1) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items*feature_dims) + state_dim]
        
            
        return self.model(input_features) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
        