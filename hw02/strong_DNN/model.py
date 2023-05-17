import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out=0.1):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, drop_out=0.1):
        super(RNN, self).__init__()

        self.lstm_1 = nn.LSTM(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=hidden_layers,
                              batch_first=True,
                              dropout=drop_out,
                              bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        r_out, _ = self.lstm_1(x, None)
        out = self.out(r_out[:])
        return out


class DNN(nn.Module):
    def __init__(self, config, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(DNN, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, config['drop_out']),
            *[BasicBlock(hidden_dim, hidden_dim, config['drop_out'])
              for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            # RNN(input_dim=input_dim,
            #     output_dim=hidden_dim,
            #     hidden_layers=2,
            #     hidden_dim=624,
            #     drop_out=config['drop_out']),
            DNN(config, input_dim, output_dim, hidden_layers, hidden_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
