import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, h_layer_size):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, h_layer_size, bias=True)
        self.fc2 = nn.Linear(h_layer_size, 1, bias=True)

        # Underfit
        # self.fc2 = nn.Linear(input_size, 1, bias=True)

        # Overfit
        # self.fc1 = nn.Linear(input_size, 64, bias=True)
        # self.fc2 = nn.Linear(64, 64, bias=True)
        # self.fc3 = nn.Linear(64, 64, bias=True)
        # self.fc4 = nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        # Underfit
        # x = F.sigmoid(self.fc2(x))

        # Overfit
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.sigmoid(self.fc4(x))

        return x
