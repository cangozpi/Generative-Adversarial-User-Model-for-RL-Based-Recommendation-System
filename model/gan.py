import torch
import numpy as np
# import custom models
from historyLSTM import History_LSTM
from generator import Generator_UserModel
from discriminator import Discriminator_RewardModel

# Note that GAN is a model which orchestrated the mini-max game (training) between the  discriminator and the  generator model.
class GAN():
    def __init__(self):        
        self.history_LSTM = History_LSTM() #TODO: pass constructor parameters when the model
        self.generator_UserModel = Generator_UserModel() #TODO: pass constructor parameters when the model
        self.discriminator_RewardModel = Discriminator_RewardModel() #TODO: pass constructor parameters when the model
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        pass

    
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
    

        discriminator_optimizer = torch.nn.optim.Adam(self.discriminator_RewardModel.parameters(), lr=0.0006,betas=[0.3,0.999])
        generator_optimizer = torch.nn.optim.Adam(self.generator_UserModel.parameters(), lr=0.0003,betas=[0.3,0.999])

        
        iteration = 1
        epochs = 150 

        # Initialize empty lists to hold d_fake, d_real and
        # the generator losses
        """ YOUR CODE HERE """
        dfake_losses = []
        dreal_losses = []
        generator_losses = []


        for epoch in range(epochs): 
            for real_click_history, display_set, clicked_items  in train_loader:
                # real_click_history --> [batch_size (#users), num_time_steps, feature_dim]
                # display_set --> [batch_size (#users), num_time_steps, num_displayed_item, feature_dim]
                # clicked_items --> [batch_size (#users), num_time_steps] display set index of the clicked item
                
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
                discriminator_optimizer.zero_grad()

                # Obtain state representations given the real user's past click history
                real_states = self.history_LSTM(real_click_history) # --> [batch_size (#users), num_time_steps, state_dim]
                # Calculate the rewards for all of the possible actions (items in the (display_set+1))
                dreal_reward = self.discriminator_RewardModel.forward(real_states, display_set) # --> [batch_size (#users), num_time_steps, (num_displayed_items+1)]
                
                # Calculate the rewards for the real user actions by masking by the actions taken by the real user
                clicked_item_mask = torch.nn.functional.one_hot(clicked_items, num_classes= ((clicked_items.shape[1])+1)).squeeze(-2) # --> [batch_size (#users), num_time_steps]
                gt_reward = dreal_reward * clicked_item_mask
                dreal_loss = torch.sum(gt_reward) # total loss/rewards for the real user actions (gt)



                # ========== generator_UserModel Loss Calculation below: 
                # Obtain state representations given the real user's past click history
                generated_actions = self.generator_UserModel(real_states) # --> [batch_size (#users), num_displayed_items]
                
                for i, action in enumerate(generated_actions[:,]):  # [batch_sizes,num_time_step,feature_dim]
                    

                       
                

                    #TODO prepare generated action sequences:
                    generated_data = []

                    fake_states = self.History_LSTM(generated_data)

                    dfake_out = self.discriminator_RewardModel.get_index(fake_images)

                    dfake_loss = 0.5 * torch.mean((dfake_out)**2)




                # ============ Total loss backpropagation:
                d_loss = dreal_loss + dfake_loss
                d_loss.backward()
                discriminator_optimizer.step()

                # Updating the generator, here is a pseudocode 
                # call zero grad
                # create 32x100 noise vector
                # generate fake samples
                # pass the fake images through D
                # calculate generator loss
                # call backward and take optimizer step
                generator_optimizer.zero_grad()
                z = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.)).sample([32,100]).to(self.device)
                
                fake_images = self.generator_UserModel.forward(z)
                dfake_out = self.discriminator_RewardModel.forward(fake_images)

                g_loss = torch.mean((dfake_out - 1)**2)

                g_loss.backward()

                generator_optimizer.step()

                if iteration % 200 == 0:
                    # Append losses to you lists (d_real, d_fake, g_loss)
                    # You may also wish to print them here for logging purposes
                    dreal_losses.append(dreal_loss.detach().cpu().numpy())
                    dfake_losses.append(dfake_loss.detach().cpu().numpy())
                    generator_losses.append(g_loss.detach().cpu().numpy())

                iteration += 1

        # Return the loss lists
        return dreal_losses, dfake_losses, generator_losses


    def test(self):
        pass #TODO: fill in this method




## ========================================================== DEBUG 
if __name__ == "__main__":
    gan = GAN() #TODO: pass in constructor parameters
    gan.gan_training_loop()