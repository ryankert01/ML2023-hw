from model import Classifier
from preprocess import same_seeds


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc


def trainer(train_set, val_set, config, device):
    # get dataloader
    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False)

    print(f'DEVICE: {device}')

    # fix random seed
    same_seeds(config['seed'])

    # create model, define a loss function, and optimizer
    model = Classifier(config=config, input_dim=config['input_dim'], hidden_layers=config['hidden_layers'],
                       hidden_dim=config['hidden_dim']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    writer = SummaryWriter()

    # train
    best_acc = 0.0
    sequencial_not_improve = 0
    for epoch in range(config['num_epoch']):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get the index of the class with the highest probability
            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # validation
        if len(val_set) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    # get the index of the class with the highest probability
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config['num_epoch'], train_acc/len(train_set), train_loss/len(
                        train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ))

                writer.add_scalar('Loss/train', train_loss /
                                  len(train_loader), epoch)
                writer.add_scalar('Loss/valid', val_loss /
                                  len(val_loader), epoch)

                writer.add_scalar('Acc/train', train_acc/len(train_set), epoch)
                writer.add_scalar('Acc/valid', val_acc/len(val_set), epoch)

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config['model_path'])
                    print('saving model with acc {:.3f}'.format(
                        best_acc/len(val_set)))
                    sequencial_not_improve = 0
                else:
                    sequencial_not_improve += 1
                    if sequencial_not_improve >= config['early_stop']:
                        break
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config['num_epoch'], train_acc /
                len(train_set), train_loss/len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), config['model_path'])
        print('saving model at last epoch')

    # delete
    del train_loader, val_loader
    gc.collect()

    return best_acc / len(val_set)
