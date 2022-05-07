import torch
import torch.nn as nn   
from torch.utils.data import DataLoader
import numpy as np
import pickle
import datetime
import itertools
import os

class Dataset(nn.Module):
    def __init__(self, data_folder, dset, split="train"):
        """
        Inputs:
            data_folder (str): location of the datasset folder.
            dset (str): type of the dataset to be used. Can be "yelp", "rsc", "tb"
            split (str): can be "train", "validation", or "test". Determines the returned dataset split. 
        """
        assert split in ["train", "test", "validation"]
        data_folder = "./dropbox"
        dset = "yelp" # choose rsc, tb, or yelp

        data_filename = os.path.join(data_folder, dset+'.pkl')
        f = open(data_filename, 'rb')
        data_behavior = pickle.load(f)
        item_features = pickle.load(f)
        f.close()
        
        # Load user splits
        filename = os.path.join(data_folder, dset+'-split.pkl')
        pkl_file = open(filename, 'rb')
        train_users = pickle.load(pkl_file)
        val_users = pickle.load(pkl_file)
        test_users = pickle.load(pkl_file)
        pkl_file.close()

        # data_behavior[user][0] is user_id
        # data_behavior[user][1][t] is displayed list at time t
        # data_behavior[user][2][t] is picked id at time t

        num_items = len(item_features[0])

        self.picked_item_features_per_user = [] # --> [user, num_time_steps, picked_item_features]
        
        users = []

        if split == "train":
            users = train_users
            
        elif split == "validation":
            users = val_users
        else: # test split
            users = test_users

        for u in users:
            picked_item_features = [] # --> [time,features]
            for picked_item_id in data_behavior[u][2]:
                
                picked_item_features.append(item_features[picked_item_id])
           
            self.picked_item_features_per_user.append(picked_item_features)
        
        

        

    def __getitem__(self, index):
        """
        Returns: tuple of (vector, vector_length)
            vector: list of feature vectors of the picked items at every time. Has shape = [num_time_steps, feature_dim]
            vector_length: length of the sequence (i.e. how many feature vectors are present)
         
        """
        list_of_picked_item_features = self.picked_item_features_per_user[index] # --> [num_time_steps, picked_item_features]
        
        return list_of_picked_item_features


    def __len__(self):
        return len(self.picked_item_features_per_user) # = user



def custom_collate_fn(data):
    """
        Used to create batches with variable sequence lengths. Output will be fed into LSTM layer.
        --
        Inputs: list(tuple of (vector, vector_length))
            vector: list of feature vectors of the picked items at every time. Has shape = [num_time_steps, feature_dim]
            vector_length: length of the sequence (i.e. how many feature vectors are present)                
    """
    # pack Sequences here for LSTM batches with padded dimensions
    length_vector = []
    for d in data:
        length_vector.append(len(d))
    max_len = max(length_vector)
    for d in data:
        temp = np.zeros((804,max_len))
        temp[:len(d)] = d
    return torch.nn.utils.rnn.pack_padded_sequence(torch.tensor(data), [length_vector], batch_first=True)
    


# ==============================================================0
if __name__ == "__main__":
    # data_folder = "./dropbox"
    # dset = "yelp" # choose rsc, tb, or yelp

    # data_filename = os.path.join(data_folder, dset+'.pkl')
    # f = open(data_filename, 'rb')
    # data_behavior = pickle.load(f)
    # item_feature = pickle.load(f)
    # f.close()
    # # data_behavior[user][0] is user_id
    # # data_behavior[user][1][t] is displayed list at time t
    # # data_behavior[user][2][t] is picked id at time t
    # size_item = len(item_feature)
    # size_user = len(data_behavior)
    # f_dim = len(item_feature[0])

    # # Load user splits
    # filename = os.path.join(data_folder, dset+'-split.pkl')
    # pkl_file = open(filename, 'rb')
    # train_user = pickle.load(pkl_file)
    # vali_user = pickle.load(pkl_file)
    # test_user = pickle.load(pkl_file)
    # pkl_file.close()

    # print("=================================================")
    # print("size_item: ", size_item, ", size_user: ", size_user, ",data_behavior: ", np.asarray(data_behavior).shape,\
    #     ",item_feature: ", np.asarray(item_feature).shape, ",f_dim: ", f_dim, \
    #         ",train_user: ", np.asarray(train_user).shape, ", vali_user: ", np.asarray(vali_user).shape, \
    #             ", test_user: ", np.asarray(test_user).shape
    #     )
    # print("=================================================")
    # # print(np.asarray(data_behavior)[14])

    # Test our custom Dataset
    data_folder = "./dropbox"
    available_datasets = ["yelp", "rsc", "tb"]
    dset = available_datasets[0] # choose rsc, tb, or yelp
    
    # Initialize Dataloaders
    train_dataset = Dataset(data_folder, dset, split="train")
    # val_dataset = Dataset(data_folder, dset, split="validation")
    # test_dataset = Dataset(data_folder, dset, split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=custom_collate_fn, drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=custom_collate_fn, drop_last=True)

    print("Dataloaders successfully instantiated !")
    
    # print(train_dataloader.dataset.shape) # = 703
    for x in train_dataloader:
        print(type(x))
        print(x)
        break
        
