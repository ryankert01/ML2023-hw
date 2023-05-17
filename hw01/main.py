from torch.utils.data import DataLoader
import pandas as pd
import torch

# file
from model import My_Model
from functions import same_seed, train_valid_split, predict, save_pred
from dataset import COVID19Dataset
from feature import select_feat
from train import trainer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
config = {
    'seed': 52011314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 10000,     # Number of epochs.
    'batch_size': 128,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    # If model has not improved for this many consecutive epochs, stop training.
    'early_stop': 500,
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    'num_of_features': 20,
    'layer_dim': 700,
    'drop_out': 0.3,
}


"""# Dataloader
Read data from files and set up training, validation, and testing sets. You do not need to modify this part.
"""

# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv(
    './covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(
    train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(
    config, train_data, valid_data, test_data)

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
    COVID19Dataset(x_valid, y_valid), \
    COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(
    valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

"""# Start training!"""
# put your model and data on the same computation device.


def training():
    model = My_Model(input_dim=x_train.shape[1], config=config).to(device)
    trainer(train_loader, valid_loader, model, config, device)


training()

"""# Plot learning curves with `tensorboard` (optional)

`tensorboard` is a tool that allows you to visualize your training progress.

If this block does not display your learning curve, please wait for few minutes, and re-run this block. It might take some time to load your logging information. 
"""

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir=./runs/

"""# Testing
The predictions of your model on testing set will be stored at `pred.csv`.
"""


model = My_Model(input_dim=x_train.shape[1], config=config).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')

"""# Reference
This notebook uses code written by Heng-Jui Chang @ NTUEE (https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
"""
