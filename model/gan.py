from cProfile import label
import torch
# import custom models
from model.historyLSTM import History_LSTM
from model.generator import Generator_UserModel
from model.discriminator import Discriminator_RewardModel
import matplotlib.pyplot as plt

def plot_results(dreal_losses, dfake_losses, val_dreal_losses, val_dfake_losses):
    plt.figure()
    plt.plot(list(range(1, len(dreal_losses)+1)), dreal_losses, marker='o', label="dreal_loss")
    plt.plot(list(range(1, len(dfake_losses)+1)), dfake_losses, marker='*', label="dfake_loss")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("results/Training Losses Plot")

    plt.figure()
    plt.plot(list(range(1, len(val_dreal_losses)+1)), val_dreal_losses, marker='o', label="dreal_loss")
    plt.plot(list(range(1, len(val_dfake_losses)+1)), val_dfake_losses, marker='*', label="dfake_loss")
    plt.title("Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("results/Validation Losses Plot")

    plt.show()

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        history_LSTM_optimizer = torch.optim.Adam(self.history_LSTM.parameters(), lr=self.lr, betas=self.betas)
        discriminator_optimizer = torch.optim.Adam(self.discriminator_RewardModel.parameters(), lr=self.lr, betas=self.betas)
        generator_optimizer = torch.optim.Adam(self.generator_UserModel.parameters(), lr=self.lr, betas=self.betas)


        # Initialize empty lists to hold the generator and discriminator losses
        dfake_losses = [] # training losses
        dreal_losses = []
        val_dfake_losses = [] # validation losses
        val_dreal_losses = []

        print("*" * 30)
        print("Training GAN Model")
        print("*" * 30)

        for epoch in range(self.epochs): 
            cur_dreal_loss = 0 # total loss for cur batch
            cur_dfake_loss = 0 # total loss for cur batch
            for real_click_history, display_set, clicked_items  in train_loader:
                # real_click_history --> [max(num_time_steps), feature_dim]
                # display_set --> [max(num_time_steps), num_displayed_item, feature_dim]
                # clicked_items --> [max(num_time_steps)] display set index of the clicked items by the real user (gt user actions)
                 
                real_click_history = real_click_history.to(self.device)
                display_set = display_set.to(self.device)
                clicked_items = clicked_items.to(self.device)

                # Updating the discriminator, here is a pseudocode        
                # call zero grad
                # pass the real actions through D
                # calculate d_real loss
                # generate fake user actions
                # pass the generated_user_actions through D
                # calculate d_fake loss
                # sum the two losses
                # call backward and take optimizer step



                # ************************************ discriminator_RewardModel Loss Calculation below: ************************************

                # Obtain state representations given the real user's past click history
                real_states = self.history_LSTM(real_click_history) # --> [batch_size (#users)=1, num_time_steps, state_dim]
                # Calculate the rewards for all of the possible actions (items in the (display_set+1))
                dreal_reward = self.discriminator_RewardModel.forward(real_states, display_set) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                
                # Calculate the rewards for the real user actions by masking by the actions taken by the real user
                class_num = ((display_set.data.shape[1])+1) # (num_displayed_items+1)
                clicked_items_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(clicked_items, batch_first=True)
                clicked_item_mask = torch.nn.functional.one_hot(clicked_items_unpacked.long(), num_classes= class_num) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                gt_reward = dreal_reward * clicked_item_mask.float()
                _, total_unpadded_num_time_steps = torch.nn.utils.rnn.pad_packed_sequence(real_click_history, batch_first=True)
                dreal_loss = torch.sum(gt_reward) / sum(total_unpadded_num_time_steps) # avg loss/rewards for the real user actions (gt)



                # ========== generator_UserModel Loss Calculation below: 
                # Obtain generated user action's indices/feature vectors for 1 time step ahead given the past real users state representation
                with torch.no_grad():
                    generated_action_indices , generated_action_vectors = self.generator_UserModel.generate_actions(real_states, display_set)  # --> [batch_size (#users), num_time_steps] , [batch_size (#users), num_time_steps, feature_dims]
                # convert rnn.PackedSequence to Tensor
                real_click_history_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(real_click_history, batch_first=True)
                # generated_action_vectors --> [batch_size (#users), num_time_steps, feature_dims]
                gen_reward = torch.tensor(0).float().to(self.device)
                for b in range(generated_action_vectors.shape[0]): # index on batch_size
                    for t in range(1, generated_action_vectors.shape[1]): # index on num_time_steps (L)
                        cur_generated_action_vector = generated_action_vectors[b, t, :].to(self.device) # --> [feature_dim]
                        cur_real_past_actions = real_click_history_unpacked[b, :t, :].to(self.device) # --> [t, feature_dim]
                        # append generated action to past history from the real user
                        cur_generated_action_with_history = torch.cat((cur_real_past_actions, cur_generated_action_vector.unsqueeze(0)), dim=0) # --> [t+1, feature_dim]
                        cur_generated_action_with_history = cur_generated_action_with_history.unsqueeze(0) # --> [1, t+1, feature_dim]
                        # obtain new state representations after taking the current generated action
                        cur_fake_state = self.history_LSTM(cur_generated_action_with_history) # --> [1, t+1, state_dim]

                        # calculate the reward for the currently generated action
                        display_set_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(display_set, batch_first=True)
                        cur_display_set = display_set_unpacked[b, :t+1, :, :].unsqueeze(0) # --> [1, t+1, num_displayed_item, feature_dim]
                        cur_dfake_reward = self.discriminator_RewardModel(cur_fake_state, cur_display_set) # --> [1, t+1, (num_displayed_items+1)]

                        # Calculate the rewards for the generated user actions by masking by the generated rewards for all of the possible acitons in the display_set
                        cur_generated_action_indices = generated_action_indices[b, :t+1].unsqueeze(0) # --> [1, t+1]
                        
                        class_num = ((display_set.data.shape[1])+1) # (num_displayed_items+1)
                        cur_clicked_item_mask = torch.nn.functional.one_hot(cur_generated_action_indices, num_classes= class_num) # --> [1, t+1, (num_displayed_items+1)]
                        
                        cur_gen_reward = cur_dfake_reward * cur_clicked_item_mask.float() # --> [1, t+1, (num_displayed_items+1)]
                        gen_reward += torch.sum(cur_gen_reward) / cur_gen_reward.shape[1]
                
                dfake_loss = gen_reward # total loss/rewards for the real user actions (gt)

                # Update Disciriminator (Reward) model
                # ============ loss backpropagation:
                combined_loss = dfake_loss - dreal_loss
                if combined_loss.requires_grad:
                    # Backprop discriminator_RewardModel
                    # Note that discriminator_RewardModel tries to minimize the combined_loss
                    for param in self.discriminator_RewardModel.parameters():
                        param.requires_grad = True
                    for param in self.generator_UserModel.parameters():
                        param.requires_grad = False
                    history_LSTM_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()
                    combined_loss.backward()
                    history_LSTM_optimizer.step()
                    discriminator_optimizer.step()




                # ************************************ generator_UserModel Loss Calculation below: ************************************
                # Obtain generated user action's indices/feature vectors for 1 time step ahead given the past real users state representation
                generated_action_indices , generated_action_vectors = self.generator_UserModel.generate_actions(real_states, display_set)  # --> [batch_size (#users), num_time_steps] , [batch_size (#users), num_time_steps, feature_dims]
                # convert rnn.PackedSequence to Tensor
                # generated_action_vectors --> [batch_size (#users), num_time_steps, feature_dims]
                gen_reward = torch.tensor(0).float().to(self.device)
                for b in range(generated_action_vectors.shape[0]): # index on batch_size
                    for t in range(1, generated_action_vectors.shape[1]): # index on num_time_steps (L)
                        cur_generated_action_vector = generated_action_vectors[b, t, :].to(self.device) # --> [feature_dim]
                        cur_real_past_actions = real_click_history_unpacked[b, :t, :].to(self.device) # --> [t, feature_dim]
                        # append generated action to past history from the real user
                        cur_generated_action_with_history = torch.cat((cur_real_past_actions, cur_generated_action_vector.unsqueeze(0)), dim=0) # --> [t+1, feature_dim]
                        cur_generated_action_with_history = cur_generated_action_with_history.unsqueeze(0) # --> [1, t+1, feature_dim]
                        # obtain new state representations after taking the current generated action
                        cur_fake_state = self.history_LSTM(cur_generated_action_with_history) # --> [1, t+1, state_dim]
                        # calculate the reward for the currently generated action
                        display_set_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(display_set, batch_first=True)
                        cur_display_set = display_set_unpacked[b, :t+1, :, :].unsqueeze(0) # --> [1, t+1, num_displayed_item, feature_dim]
                        
                        cur_dfake_reward = self.discriminator_RewardModel(cur_fake_state, cur_display_set) # --> [1, t+1, (num_displayed_items+1)]

                        # Calculate the rewards for the generated user actions by masking by the generated rewards for all of the possible acitons in the display_set
                        cur_generated_action_indices = generated_action_indices[b, :t+1].unsqueeze(0) # --> [1, t+1]
                        
                        class_num = ((display_set.data.shape[1])+1) # (num_displayed_items+1)
                        clicked_item_mask = torch.nn.functional.one_hot(clicked_items.long().data, num_classes= class_num) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                        cur_clicked_item_mask = torch.nn.functional.one_hot(cur_generated_action_indices, num_classes= class_num) # --> [1, t+1, (num_displayed_items+1)]
                        
                        cur_gen_reward = cur_dfake_reward * cur_clicked_item_mask.float() # --> [1, t+1, (num_displayed_items+1)]
                        gen_reward += torch.sum(cur_gen_reward) / cur_gen_reward.shape[1]
                
                dfake_loss = -1 * gen_reward # total loss/rewards for the real user actions (gt)
                
                # ============ loss backpropagation:
                combined_loss = dfake_loss
                if combined_loss.requires_grad:
                    # backprop generator_UserModel
                    # Note that generator_UserModel tries to maximize the combined_loss
                    for param in self.generator_UserModel.parameters():
                        param.requires_grad = True
                    for param in self.discriminator_RewardModel.parameters():
                        param.requires_grad = False
                    history_LSTM_optimizer.zero_grad()
                    generator_optimizer.zero_grad()
                    discriminator_optimizer.zero_grad()
                    combined_loss.backward()
                    history_LSTM_optimizer.step()
                    generator_optimizer.step()

                    # record losses
                    cur_dfake_loss += dreal_loss.detach().cpu().numpy()
                    cur_dreal_loss += dfake_loss.detach().cpu().numpy()

            # logging
            dreal_losses.append(cur_dfake_loss)
            dfake_losses.append(cur_dreal_loss)

        

            # ================== Validation part
            val_cur_dreal_loss = 0 # total loss for cur batch
            val_cur_dfake_loss = 0 # total loss for cur batch
            for real_click_history, display_set, clicked_items  in validation_loader:
                # real_click_history --> [max(num_time_steps), feature_dim]
                # display_set --> [max(num_time_steps), num_displayed_item, feature_dim]
                # clicked_items --> [max(num_time_steps)] display set index of the clicked items by the real user (gt user actions)
                
                real_click_history = real_click_history.to(self.device)
                display_set = display_set.to(self.device)
                clicked_items = clicked_items.to(self.device)

                with torch.no_grad():
                     # Obtain state representations given the real user's past click history
                    real_states = self.history_LSTM(real_click_history) # --> [batch_size (#users)=1, num_time_steps, state_dim]
                    # Calculate the rewards for all of the possible actions (items in the (display_set+1))
                    dreal_reward = self.discriminator_RewardModel.forward(real_states, display_set) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                    
                    # Calculate the rewards for the real user actions by masking by the actions taken by the real user
                    class_num = ((display_set.data.shape[1])+1) # (num_displayed_items+1)
                    clicked_items_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(clicked_items, batch_first=True)
                    clicked_item_mask = torch.nn.functional.one_hot(clicked_items_unpacked.long(), num_classes= class_num) # --> [batch_size (#users), max(num_time_steps), (num_displayed_items+1)]
                    gt_reward = dreal_reward * clicked_item_mask.float()
                    dreal_loss = torch.sum(gt_reward) / dreal_reward.shape[1] # avg loss/rewards for the real user actions (gt)



                    # ========== generator_UserModel Loss Calculation below: 
                    # Obtain generated user action's indices/feature vectors for 1 time step ahead given the past real users state representation
                    generated_action_indices , generated_action_vectors = self.generator_UserModel.generate_actions(real_states, display_set)  # --> [batch_size (#users), num_time_steps] , [batch_size (#users), num_time_steps, feature_dims]
                    # convert rnn.PackedSequence to Tensor
                    real_click_history_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(real_click_history, batch_first=True)
                    # generated_action_vectors --> [batch_size (#users), num_time_steps, feature_dims]
                    gen_reward = torch.tensor(0).float().to(self.device)
                    for b in range(generated_action_vectors.shape[0]): # index on batch_size
                        for t in range(1, generated_action_vectors.shape[1]): # index on num_time_steps (L)
                            cur_generated_action_vector = generated_action_vectors[b, t, :].to(self.device) # --> [feature_dim]
                            cur_real_past_actions = real_click_history_unpacked[b, :t, :].to(self.device) # --> [t, feature_dim]
                            # append generated action to past history from the real user
                            cur_generated_action_with_history = torch.cat((cur_real_past_actions, cur_generated_action_vector.unsqueeze(0)), dim=0) # --> [t+1, feature_dim]
                            cur_generated_action_with_history = cur_generated_action_with_history.unsqueeze(0) # --> [1, t+1, feature_dim]
                            # obtain new state representations after taking the current generated action
                            cur_fake_state = self.history_LSTM(cur_generated_action_with_history) # --> [1, t+1, state_dim]
                            # calculate the reward for the currently generated action
                            display_set_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(display_set, batch_first=True)
                            cur_display_set = display_set_unpacked[b, :t+1, :, :].unsqueeze(0) # --> [1, t+1, num_displayed_item, feature_dim]
                            cur_dfake_reward = self.discriminator_RewardModel(cur_fake_state, cur_display_set) # --> [1, t+1, (num_displayed_items+1)]

                            # Calculate the rewards for the generated user actions by masking by the generated rewards for all of the possible acitons in the display_set
                            cur_generated_action_indices = generated_action_indices[b, :t+1].unsqueeze(0) # --> [1, t+1]
                            
                            class_num = ((display_set.data.shape[1])+1) # (num_displayed_items+1)
                            cur_clicked_item_mask = torch.nn.functional.one_hot(cur_generated_action_indices, num_classes= class_num) # --> [1, t+1, (num_displayed_items+1)]
                            
                            cur_gen_reward = cur_dfake_reward * cur_clicked_item_mask.float() # --> [1, t+1, (num_displayed_items+1)]
                            gen_reward += torch.sum(cur_gen_reward) / cur_gen_reward.shape[1]
                    
                    dfake_loss = gen_reward # total loss/rewards for the real user actions (gt)

                    # record losses
                    val_cur_dfake_loss += dreal_loss.detach().cpu().numpy()
                    val_cur_dreal_loss += dfake_loss.detach().cpu().numpy()

            # logging
            val_dreal_losses.append(val_cur_dfake_loss)
            val_dfake_losses.append(val_cur_dreal_loss)

            print("_" * 25)
            print(f"epoch: [{epoch+1}/{self.epochs}], train_dreal_loss: {dreal_losses[-1]}, train_dfake_loss: {dfake_losses[-1]} \
                val_dreal_loss: {val_dreal_losses[-1]}, val_dfake_loss: {val_dfake_losses[-1]}")
            print("_" * 25)

        plot_results(dreal_losses, dfake_losses, val_dreal_losses, val_dfake_losses)
        # Return the losses
        return dreal_losses, dfake_losses, val_dreal_losses, val_dfake_losses


    def test(self):
        pass #TODO: fill in this method




## ========================================================== DEBUG 
if __name__ == "__main__":
    gan = GAN() #TODO: pass in constructor parameters
    gan.gan_training_loop(train_loader, validation_loader)
    # gan.test() #TODO