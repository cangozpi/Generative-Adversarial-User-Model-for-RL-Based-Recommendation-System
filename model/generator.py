import torch
from torch import nn

# Note that User Model is the Generator in this context
class Generator_UserModel(nn.Module):
    def __init__(self, input_size, output_size, n_hidden, hidden_dim):
        """
        input_size: equals ((num_displayed_items+1)*feature_dims + state_dim)
        output_size: equals (num_displayed_items+1)
        """
        
        super().__init__()
        self.input_size = input_size
        layers = []
        
        layers.extend([torch.nn.Linear(input_size, hidden_dim),torch.nn.ReLU()])
        
        for n in range(n_hidden-1):
            layers.extend([torch.nn.Linear(hidden_dim, hidden_dim),torch.nn.ReLU()])
            
            
        # Classification Layer (outputs score for each display_item)
        layers.extend[[torch.nn.Linear(hidden_dim, output_size), torch.nn.Tanh()]]
         
        self.model = torch.nn.Sequential(*layers) # (inp_0 inp_1 .. inp_k) --> classification(inp_0, inp_1, .. inp_k) 
                                                

    def forward(self, state, displayed_items):
        """
        Input:
            state (torch.Tensor): [batch_size (#users), num_time_steps, state_dim]
            displayed_items (torch.Tensor): [batch_size (#users), num_time_steps, num_displayed_items, feature_dims]
        Return:
            action_scores (torch.Tensor): [batch_size (#users), num_time_steps, num_displayed_items]
        """
        # Prepare input
        batch_size = state.shape[0]
        num_time_steps = state.shape[1]
        # concat zero vector to displayed items to represent user not clicking on any of the displayed items
        not_clicking_feature_vec = torch.zeros((1, displayed_items.shape[-1])) # --> [1, feature_dims]
        displayed_items = torch.cat((displayed_items, not_clicking_feature_vec), -2) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1), feature_dims]
        displayed_items_flat = displayed_items.view(batch_size, num_time_steps,-1) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)*feature_dims]
        input_features = torch.cat((displayed_items_flat, state), dim=-1) # --> [batch_size (#users), num_time_steps, ((num_displayed_items+1)*feature_dims + state_dim)]
        
        out = self.model(input_features) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)]
        
        return out


    def get_index(self, state, displayed_items):
        """
        Input:
            state (torch.Tensor): [batch_size (#users), num_time_steps, state_dim]
            displayed_items (torch.Tensor): [batch_size (#users), num_time_steps, num_displayed_items, feature_dims]
        Return:
            generated_action_indices (torch.Tensor): [batch_size (#users), num_time_steps] indices of the actions chosen from the displayed_items by the user model
        """
        out = self.forward(state, displayed_items) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)]
        # find the action with the highest probability
        pred_probs = torch.nn.functional.softmax(out, dim=2) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)]
        generated_action_indices = torch.argmax(pred_probs, dim=2) # --> [batch_size (#users), num_time_steps]
        
        return generated_action_indices
        
        
        