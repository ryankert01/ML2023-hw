from preprocess import pre_parse_data
from train import trainer
from test import tester

import torch
import torch.nn.functional

import gc

# hyperparameters

config = {
    # data prarameters
    # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
    # 29 is optimal ?! but facing memories allocation problem
    'concat_nframes': 25,
    'train_ratio': 0.8,

    # training parameters
    'seed': 5201314,                    # random seed
    'batch_size': 256,                  # batch size
    'num_epoch': 60,                   # the number of training epoch
    'learning_rate': 1e-4,              # learning rate
    'model_path': './model.ckpt',       # the path where the checkpoint will be saved
    'early_stop': 3,                    # stop after not improving _ times

    # model parameters
    # the input dim of the model, you should not change the value
    'input_dim': 39 * 25,
    'hidden_layers': 5,                 # the number of hidden layers
    'hidden_dim': 1700,                 # the hidden dim
    'drop_out': 0.25,                    # drop out rate

}

config['input_dim'] = 39 * config['concat_nframes']

# define device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# pre-process data
train_set, val_set = pre_parse_data(config=config)

# training
trainer(train_set, val_set, config=config, device=device)

del train_set, val_set
gc.collect()

# testing
tester(config=config, device=device)
