import torch
from torch import nn
# import custom models
from historyLSTM import History_LSTM
from generator import Generator_UserModel
from discriminator import Discriminator_RewardModel

# Note that GAN is a model which orchestrated the mini-max game (training) between the  discriminator and the  generator model.
class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_LSTM = History_LSTM() #TODO: pass constructor parameters when the model
        self.generator_UserModel = Generator_UserModel() #TODO: pass constructor parameters when the model
        self.discriminator_RewardModel = Discriminator_RewardModel() #TODO: pass constructor parameters when the model
        pass

    def forward(self, data):
        """
        Input:
            data (torch.Tensor): DataLoader's batched data. 
        Return:
            generated_actions (torch.tensor): Actions taken by the generator_UserModel
            UserModel_rewards (torch.tensor): Reward values for the generator_UserModel generated actions.
            ground_truth_rewards (torch.tensor): Reward values for the ground truth actions.
        """
        initial_state = None # initialize vector representation of the initial state

        pass

    def loss(self):
        pass