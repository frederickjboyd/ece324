import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(10 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # No batch normalization
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ReallySmallNet(nn.Module):
    def __init__(self):
        super(ReallySmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(10 * 1 * 1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BatchNorm(nn.Module):
    def __init__(self):
        super(BatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv_bn = nn.BatchNorm2d(10)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(10 * 12 * 12, 32)
        self.fc1_bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # Batch normalization
        x = self.pool(F.relu(self.conv_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv_bn(self.conv2(x))))
        x = x.view(-1, 10 * 12 * 12)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)

        return x


class FourConvLayers(nn.Module):
    def __init__(self):
        super(FourConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(10 * 1 * 1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # print(x.size())
        x = x.view(-1, 10 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
