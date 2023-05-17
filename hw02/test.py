from model import Classifier
from preprocess import preprocess_data
from dataset import LibriDataset

import torch
from torch.utils.data import DataLoader
import torch.nn.functional
import numpy as np
from tqdm import tqdm


def tester(config, device):
    # load data
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat',
                             phone_path='./libriphone', concat_nframes=config['concat_nframes'])
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(
        test_set, batch_size=config['batch_size'], shuffle=False)

    # load model
    model = Classifier(config=config, input_dim=config['input_dim'], hidden_layers=config['hidden_layers'],
                       hidden_dim=config['hidden_dim']).to(device)
    model.load_state_dict(torch.load(config['model_path']))

    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)

            # get the index of the class with the highest probability
            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
