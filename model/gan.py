import torch
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
        pass

    
    def gan_training_loop(train_loader, test_loader):
        """
        Input:
            train_loader (torch.Tensor): training DataLoader
            test_loader (torch.Tensor): training DataLoader 
        Return:
            generated_actions (torch.tensor): Actions taken by the generator_UserModel.
            UserModel_rewards (torch.tensor): Reward values for the generator_UserModel generated actions.
            ground_truth_rewards (torch.tensor): Reward values for the ground truth actions.
        """
    

        discriminator_optimizer = torch.nn.optim.Adam(D.parameters(), lr=0.0006,betas=[0.3,0.999])
        generator_optimizer = torch.nn.optim.Adam(G.parameters(), lr=0.0003,betas=[0.3,0.999])

        

        # We created a fixed noise vector, we will use this noise to visualize generated samples
        # throughout the training. 
        fixed_noise = (torch.rand(100, 100) * 2 - 1).cuda()
        iteration = 1
        epochs = 150 

        # Initialize empty lists to hold d_fake, d_real and
        # the generator losses
        """ YOUR CODE HERE """
        dfake_losses = []
        dreal_losses = []
        generator_losses = []


        for epoch in range(epochs): 
            for real_data, display_set, clicked_ids  in train_loader:
                real_data = real_data.cuda()
                display_set = display_set.cuda()

                # Updating the discriminator, here is a pseudocode        
                # call zero grad
                # pass the real images through D
                # calculate d_real loss
                # create 32x100 noise vector
                # generate fake samples
                # pass the fake images through D
                # calculate d_fake loss
                # sum the two losses
                # call backward and take optimizer step

                real_states = self.history_LSTM(real_data)


                discriminator_optimizer.zero_grad()
                dreal_out = D.forward(real_states,display_set)
                
                
                #TODO dreal_loss 
            
                fake_action = G.forward(real_states,display_set)
                
                #TODO prepare generated action sequences
                generated_data = []

                fake_states = self.History_LSTM(generated_data)

                dfake_out = D.forward(fake_images)

                dfake_loss = 0.5 * torch.mean((dfake_out)**2)

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
                z = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.)).sample([32,100]).cuda()
                
                fake_images = G.forward(z)
                dfake_out = D.forward(fake_images)

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