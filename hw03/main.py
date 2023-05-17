import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from torch.utils.tensorboard import SummaryWriter


from models import Classifier
from param import device, myseed, train_tfm, pre_train_tfm, n_epochs, pre_epochs, _exp_name
from train import train, test
from utils import same_seed

same_seed(myseed)

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
# model = resnet18(weights=None).to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss().to(device)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

writer = SummaryWriter()

# pre-train
train(model, criterion, optimizer, writer, pre_train_tfm, pre_epochs)

# model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
# fine-tune
# train(model, criterion, optimizer, writer, train_tfm, n_epochs, pre_epochs)

model_best = Classifier().to(device)
# model_best = resnet18(weights=None).to(device)

test(model_best)