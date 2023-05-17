import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out=0.1):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(drop_out),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()
        dropout = config['drop_out']
        self.lstm = nn.LSTM(39, 600, 3,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            # BasicBlock(800, hidden_dim, dropout),
            # *[BasicBlock(hidden_dim, hidden_dim, dropout)
            #   for _ in range(hidden_layers)],
            # BasicBlock(hidden_dim, output_dim, dropout)
            BasicBlock(1200, output_dim, dropout)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 39)
        x, _ = self.lstm(x)
        x = self.fc(x[:, x.size(1)//2, :])
        return x

# class RNN(nn.Module):
#     def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256, drop_out=0.1):
#         super(RNN, self).__init__()

#         self.lstm_1 = nn.LSTM(input_size=input_dim,
#                               hidden_size=hidden_dim,
#                               num_layers=hidden_layers,
#                               batch_first=True,
#                               dropout=drop_out,
#                               bidirectional=True)
#         self.out = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, x):
#         r_out, _ = self.lstm_1(x, None)
#         out = self.out(r_out[:])
#         return out


# class DNN(nn.Module):
#     def __init__(self, config, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
#         super(DNN, self).__init__()

#         self.fc = nn.Sequential(
#             BasicBlock(input_dim, hidden_dim, config['drop_out']),
#             *[BasicBlock(hidden_dim, hidden_dim, config['drop_out'])
#               for _ in range(hidden_layers)],
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class Classifier(nn.Module):
#     def __init__(self, config, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
#         super(Classifier, self).__init__()

#         self.fc = nn.Sequential(
#             RNN(input_dim=input_dim,
#                 output_dim=400,
#                 hidden_layers=3,
#                 hidden_dim=400,
#                 drop_out=config['drop_out']),
#             nn.BatchNorm1d(800),
#             nn.ReLU(),
#             nn.Dropout(config['drop_out']),
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x
