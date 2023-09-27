import torch.nn as nn
import torch
from convLSTM import ConvLSTM

class CNNLSTMModel(nn.Module):
    
    def __init__(self):
        super(CNNLSTMModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=30, kernel_size=(3,3,3), stride=(1,1,1)),
            nn.BatchNorm3d(num_features=30),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            nn.Conv3d(in_channels=30, out_channels=60, kernel_size=(3,3,3), stride=(1,1,1)),
            nn.BatchNorm3d(num_features=60),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            nn.Conv3d(in_channels=60, out_channels=80, kernel_size=(3,3,3), stride=(1,1,1)),
            nn.Conv3d(in_channels=80, out_channels=80, kernel_size=(3,3,3), stride=(1,1,1)),
            nn.BatchNorm3d(num_features=80),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(1,2,2)),
        )

        self.lstm = ConvLSTM(input_dim = 80,
                             hidden_dim= [256, 384], 
                             kernel_size=(3,3),
                             num_layers=2,
                             batch_first = True,
                             bias=True,
                             return_all_layers=False)

        self.pool2d = nn.MaxPool2d(kernel_size=(7,7), stride=(7,7))

        self.linear = nn.Linear(in_features=1*4*4*384, out_features=1)

    def forward(self, images):
        out = self.cnn(images)
        out = torch.transpose(out, 1, 2)
        _, last_states = self.lstm(out)
        out = last_states[0][0]
        out = torch.squeeze(out)
        out = self.pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
