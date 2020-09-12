import argparse
import numpy as np
import matplotlib.pyplot as plt
from dispkernel import dispKernel

import torch
import torch.nn as nn

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format', default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',
                    default="valid")
parser.add_argument('numtrain', help='number of training samples', type=int, default=200)
parser.add_argument('numvalid', help='number of validation samples', type=int, default=20)
parser.add_argument('-seed', help='random seed', type=int, default=1)
parser.add_argument('-learningrate', help='learning rate', type=float, default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],
                    default='linear')
parser.add_argument('-numepoch', help='number of epochs', type=int, default=50)

args = parser.parse_args()

# Load data into tensor
traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"
validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

train_data = torch.from_numpy(np.loadtxt(traindataname, dtype=np.single, delimiter=','))
train_label = torch.from_numpy(np.loadtxt(trainlabelname, dtype=np.single, delimiter=','))
valid_data = torch.from_numpy(np.loadtxt(validdataname, dtype=np.single, delimiter=','))
valid_label = torch.from_numpy(np.loadtxt(validlabelname, dtype=np.single, delimiter=','))

# Set manual torch seed
torch.manual_seed(args.seed)


class SNC(nn.Module):
    def __init__(self, params):
        self.num_train = params.numtrain
        self.num_valid = params.numvalid

        self.learning_rate = params.learningrate
        self.num_epoch = params.numepoch

        self.act_function = params.actfunction

        self.threshold = 0.5

        super(SNC, self).__init__()
        self.fc1 = nn.Linear(9, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x


def accuracy(predictions,label):
    total_corr = 0

    index = 0
    for c in predictions.flatten():
        if c.item() > 0.5:
            r = 1.0
        else:
            r = 0.0
        if r == label[index].item():
            total_corr += 1
        index += 1

    return total_corr / len(label)


smallNN = SNC(args)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(smallNN.parameters(), lr=args.learningrate)

# Arrays to keep track of accuracy and loss
training_loss = []
validation_loss = []
training_accuracy = []
validation_accuracy = []

for i in range(0, args.numepoch):
    # Set gradients to zero
    optimizer.zero_grad()

    # Make prediction using training data
    predict = smallNN(train_data)

    # Calculate loss
    loss = loss_function(predict.squeeze(), train_label.float())

    # Calculate gradients
    loss.backward()

    # Update values of weights and bias
    optimizer.step()

    # Calculate training accuracy
    train_accuracy = accuracy(predict, train_label)

    # Calculate loss and accuracy of validation data
    valid_predict = smallNN(valid_data)
    valid_loss = loss_function(valid_predict.squeeze(), valid_label.float())
    valid_accuracy = accuracy(valid_predict, valid_label)

    # Keep track of current epoch data for plotting later
    training_loss.append(loss)
    training_accuracy.append(train_accuracy)
    validation_loss.append(valid_loss)
    validation_accuracy.append(valid_accuracy)


def plot_data():
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title('Average Training & Validation Losses (Good Learning Rate)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig('loss')
    plt.show()

    plt.plot(training_accuracy)
    plt.plot(validation_accuracy)
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title('Training & Validation Accuracies (Good Learning Rate)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy')
    plt.show()


# Extract weights and bias from trained model
weights = []
bias = []
for name, param in smallNN.named_parameters():
    if name == 'fc1.weight':
        weights = param[0].tolist()
    elif name == 'fc1.bias':
        bias = param[0].tolist()

dispKernel(weights, 3, 9)
plot_data()
