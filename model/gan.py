import torch
# import custom models
from model.historyLSTM import History_LSTM
from model.generator import Generator_UserModel
from model.discriminator import Discriminator_RewardModel

# Note that GAN is a model which orchestrated the mini-max game (training) between the  discriminator and the  generator model.
class GAN():
    def __init__(self, history_input_size, history_hidden_size, history_num_layers, \
        generator_input_size, generator_output_size, generator_n_hidden, generator_hidden_dim, \
            discriminator_input_size, discriminator_output_size, discriminator_n_hidden, discriminator_hidden_dim, \
                lr=0.0006, betas=[0.3,0.999], epochs=150):        
        """
        == Parameters of the History_LSTM:
            history_input_size (int): feature_dim of the actions.
            history_hidden_size (int): dimension of the state representation vector (dim of output of the History_LSTM)
            history_num_layers (int): number of recurrent layers in the History_LSTM.

        == Parameters of the Generator_UserModel:
            generator_input_size (int): equals ((num_displayed_items+1)*feature_dims + state_dim)
            generator_output_size (int): equals (num_displayed_items+1)
            generator_n_hidden (int): number of hidden layers in the generator model.
            generator_hidden_dim (int): hidden dimension of the layers in the generator model.

        == Parameters of teh Discriminator_RewardModel:
            discriminator_input_size (int): should equal (num_displayed_items*feature_dims) + state_dim.
            discriminator_output_size (int): should equal (num_displayed_items+1). 
            discriminator_n_hidden (int): number of hidden layers of the Discriminator model's MLP.
            discriminator_hidden_dim (int): hidden dimension of the layers of the Discriminator model's MLP.
        
        == Hyperparameters of the training
            lr (int): learning rate used by the optimizer.
            betas (tuple): beta values used by the ADAM optimizer.
            epochs (int): number of epochs to train.
        """
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.history_LSTM = History_LSTM(history_input_size, history_hidden_size, history_num_layers).to(self.device)
        self.generator_UserModel = Generator_UserModel(generator_input_size, generator_output_size, generator_n_hidden, generator_hidden_dim).to(self.device)
        self.discriminator_RewardModel = Discriminator_RewardModel(discriminator_input_size, discriminator_output_size, discriminator_n_hidden, discriminator_hidden_dim).to(self.device)
        self.lr = lr
        self.betas = betas
        self.epochs = epochs
        

    
    def gan_training_loop(self, train_loader, validation_loader):
        """
        Input:
            train_loader (torch.Tensor): training DataLoader
            test_loader (torch.Tensor): training DataLoader 
        Return:
            generated_actions (torch.tensor): Actions taken by the generator_UserModel.
            UserModel_rewards (torch.tensor): Reward values for the generator_UserModel generated actions.
            ground_truth_rewards (torch.tensor): Reward values for the ground truth actions.
        """
        #TODO implement validation

        discriminator_optimizer = torch.optim.Adam(self.discriminator_RewardModel.parameters(), lr=self.lr, betas=self.betas)
        generator_optimizer = torch.optim.Adam(self.generator_UserModel.parameters(), lr=self.lr, betas=self.betas)


        # Initialize empty lists to hold the generator and discriminator losses
        dfake_losses = []
        dreal_losses = []

        iteration = 1
        for epoch in range(self.epochs): 
            for real_click_history, display_set, clicked_items  in train_loader:
                # real_click_history --> [batch_size (#users), max(num_time_steps), feature_dim]
                # display_set --> [batch_size (#users), max(num_time_steps), num_displayed_item, feature_dim]
                # clicked_items --> [batch_size (#users), max(num_time_steps)] display set index of the clicked items by the real user (gt user actions)
                
                real_click_history = real_click_history.to(self.device)
                display_set = display_set.to(self.device)

                # Updating the discriminator, here is a pseudocode        
                # call zero grad
                # pass the real actions through D
                # calculate d_real loss
                # generate fake user actions
                # pass the generated_user_actions through D
                # calculate d_fake loss
                # sum the two losses
                # call backward and take optimizer step



                # ========== discriminator_RewardModel Loss Calculation below:

                # Obtain state representations given the real user's past click history
                real_states = self.history_LSTM(real_click_history) # --> [batch_size (#users), num_time_steps, state_dim]
                # Calculate the rewards for all of the possible actions (items in the (display_set+1))
                dreal_reward = self.discriminator_RewardModel.forward(real_states, display_set) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                
                # Calculate the rewards for the real user actions by masking by the actions taken by the real user
                clicked_item_mask = torch.nn.functional.one_hot(clicked_items, num_classes= ((clicked_items.shape[1])+1)) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                gt_reward = dreal_reward * clicked_item_mask
                dreal_loss = torch.sum(gt_reward) # total loss/rewards for the real user actions (gt)



                # ========== generator_UserModel Loss Calculation below: 
                # Obtain generated user action's indices/feature vectors for 1 time step ahead given the past real users state representation
                generated_action_indices , generated_action_vectors = self.generator_UserModel.generate_actions(real_states, display_set)  #[batch_size (#users), num_time_steps, (num_displayed_items+1)] , [batch_size (#users), num_time_steps, feature_dims]
                # Obtain new state representations after taking the generated actions
                fake_states = self.history_LSTM(generated_action_vectors)
                dfake_reward = self.discriminator_RewardModel.forward(fake_states, display_set) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)]
                # Calculate the rewards for the generated user actions by masking by the generated rewards for all of the possible acitons in the display_set
                clicked_item_mask = torch.nn.functional.one_hot(generated_action_indices, num_classes= ((generated_action_indices.shape[1])+1)).squeeze(-2) # --> [batch_size (#users), num_time_steps]
                gen_reward = dfake_reward * clicked_item_mask
                dfake_loss = torch.sum(gen_reward) # total loss/rewards for the real user actions (gt)



                # ============ Total loss backpropagation:
                combined_loss = dfake_loss - dreal_loss

                # Backprop discriminator_RewardModel
                # Note that discriminator_RewardModel tries to minimize the combined_loss
                discriminator_optimizer.zero_grad()
                combined_loss.backward()
                discriminator_optimizer.step()

                # backprop generator_UserModel
                # Note that generator_UserModel tries to maximize the combined_loss
                generator_optimizer.zero_grad()
                combined_loss *= -1
                combined_loss.backward()
                generator_optimizer.step()

                # logging
                if iteration % 200 == 0:
                    # Append losses to you lists (d_real, d_fake)
                    # You may also wish to print them here for logging purposes
                    dreal_losses.append(dreal_loss.detach().cpu().numpy())
                    dfake_losses.append(dfake_loss.detach().cpu().numpy())

                iteration += 1

        # Return the losses
        return dreal_losses, dfake_losses


    def test(self):
        pass #TODO: fill in this method




## ========================================================== DEBUG 
if __name__ == "__main__":
    gan = GAN() #TODO: pass in constructor parameters
    gan.gan_training_loop(train_loader, validation_loader)
    # gan.test() #TODO