from model.gan import GAN
from data import Dataset, custom_collate_fn
import yaml
from copy import deepcopy
import argparse
from torch.utils.data import DataLoader

import torch



def arg_parse():
    parser = argparse.ArgumentParser(description='Generative Adversarial User Model.')
    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='Path of the configurations yaml file.')
    parser.add_argument('--mode', type=str, default="train",
                        help='either [\'train\', \'test\']. Specifies the mode as either training mode or testing mode.')
    parser.add_argument('--data_folder', type=str, default="./dropbox",
                        help='Path (str) that holds the dataset file.')
    parser.add_argument('--dataset', type=str, default="yelp",
                        help='either ["yelp", "rsc", "tb"]. Dataset to use for initializing the DataLoaders.')
    

    args = parser.parse_args()
    return args
    


def parse_config_yaml(file_path):
    """
    Input:
        file_path (str): path of the config yaml file
    Return:
        config_dict (dict): dictionary containing the information in the yaml file
    Given a path for the config yaml file as str, parses the parameters and returns the dictionary.
    """
    with open(file_path) as config_yaml:
        config_dict_yaml = yaml.load(config_yaml, Loader=yaml.SafeLoader)
    return config_dict_yaml


def get_dataLoaders(data_folder, dset, batch_size):
    # Initialize Dataloaders
    train_dataset = Dataset(data_folder, dset, split="train")
    val_dataset = Dataset(data_folder, dset, split="validation")
    test_dataset = Dataset(data_folder, dset, split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader



if __name__ == "__main__":
    # Parse the command line arguments
    args = arg_parse()
    # Parse the configurations yaml file
    config_dict = parse_config_yaml(args.config_path)

       # Initialize dataloaders
    data_folder = args.data_folder
    dset = args.dataset # choose rsc, tb, or yelp
    assert dset in ["yelp", "rsc", "tb"]
    train_dataloader, val_dataloader, test_dataloader = get_dataLoaders(data_folder, dset, config_dict['batch_size'])

    if args.mode == "train":
        real_click_history, display_set, clicked_items = next(iter(train_dataloader))
        display_set_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(display_set, batch_first=True)
    elif args.mode == "test":
        real_click_history, display_set, clicked_items = next(iter(test_dataloader))
        display_set_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(display_set, batch_first=True)
                        
    config_dict["generator_output_size"] = display_set_unpacked.shape[-2] + 1
    config_dict["discriminator_output_size"] = display_set_unpacked.shape[-2] + 1
    config_dict["history_input_size"] = display_set_unpacked.shape[-1]
    config_dict["generator_input_size"] = config_dict["history_hidden_size"] + (config_dict["generator_output_size"] * config_dict["history_input_size"])
    config_dict["discriminator_input_size"] = config_dict["history_hidden_size"] + (config_dict["discriminator_output_size"] * config_dict["history_input_size"])

    # Initialize the GAN model
    gan = GAN(config_dict, config_dict['history_input_size'], config_dict['history_hidden_size'], config_dict['history_num_layers'], \
        config_dict['generator_input_size'], config_dict['generator_output_size'], config_dict['generator_n_hidden'], config_dict['generator_hidden_dim'], \
            config_dict['discriminator_input_size'], config_dict['discriminator_output_size'], config_dict['discriminator_n_hidden'], config_dict['discriminator_hidden_dim'], \
                lr=config_dict['lr'], betas=config_dict['betas'], epochs=config_dict['epochs'])


    # Train/Test using the GAN model
    if args.mode == "train":
        train_dreal_losses, train_dfake_losses, val_dreal_losses, val_dfake_losses = gan.gan_training_loop(train_dataloader, val_dataloader)
    else:
        test_cur_dreal_loss, test_cur_dfake_loss = gan.test(test_dataloader)