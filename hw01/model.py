import torch.nn as nn


class My_Model(nn.Module):
    def __init__(self, input_dim, config):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        layer_dim: int = config['layer_dim']
        self.layers = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(layer_dim),
            nn.Dropout(config['drop_out']),
            nn.Linear(layer_dim, layer_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(layer_dim),
            nn.Dropout(config['drop_out']),
            nn.Linear(layer_dim, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
