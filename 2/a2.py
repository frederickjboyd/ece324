import argparse
import numpy as np
import matplotlib.pyplot as plt
from dispkernel import dispKernel

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class SingleNeuronClassifier:
    def __init__(self, params):
        traindataname = params.trainingfile + "data.csv"
        trainlabelname = params.trainingfile + "label.csv"
        validdataname = params.validationfile + "data.csv"
        validlabelname = params.validationfile + "label.csv"

        self.train_data = np.loadtxt(traindataname, delimiter=',')
        self.train_label = np.loadtxt(trainlabelname, delimiter=',')
        self.valid_data = np.loadtxt(validdataname, delimiter=',')
        self.valid_label = np.loadtxt(validlabelname, delimiter=',')

        self.num_train = params.numtrain
        self.num_valid = params.numvalid

        self.learning_rate = params.learningrate
        self.num_epoch = params.numepoch

        self.act_function = params.actfunction

        self.threshold = 0.5

        print('Seed:', params.seed)
        print('Learning rate:', self.learning_rate)
        print('Epochs:', self.num_epoch)
        print('Activation function:', self.act_function)

        # Set seed for pseudo-random number generator
        np.random.seed(params.seed)

        # Initialize weights and bias to random values
        self.weights = np.random.random(9)
        self.bias = np.random.random()

        # Arrays to plot
        self.training_error = np.zeros(self.num_epoch)
        self.validation_error = np.zeros(self.num_epoch)
        self.training_accuracy = np.zeros(self.num_epoch)
        self.validation_accuracy = np.zeros(self.num_epoch)

    def train(self):
        # Error check
        if len(self.train_data) != len(self.train_label):
            print('Error: training data has length', len(self.train_data), 'and training labels has length',
                  len(self.train_label))
            return
        for epoch in range(self.num_epoch):
            self.handle_epoch(epoch)

        print('Final training accuracy:', self.training_accuracy[len(self.training_accuracy) - 1])
        print('Final validation accuracy:', self.validation_accuracy[len(self.validation_accuracy) - 1])

    def handle_epoch(self, epoch):
        weight_gradient_accum = np.zeros(len(self.weights))
        bias_gradient_accum = 0
        mse_sum = 0

        # Keep track of correct answers to calculate accuracies
        training_correct_count = 0
        validation_correct_count = 0

        train_data_length = len(self.train_data)
        for i in range(train_data_length):
            z = np.dot(self.train_data[i], self.weights) + self.bias

            # Activation function
            y = self.apply_activation_function(z)

            # Determine if current prediction is accurate
            if (y > self.threshold and self.train_label[i] == 1) or (y <= self.threshold and self.train_label[i] == 0):
                training_correct_count += 1

            # Calculate weight gradients
            for j in range(len(self.train_data[i])):
                if self.act_function == 'sigmoid':
                    weight_gradient_accum[j] += 2 * (y - self.train_label[i]) * sigmoid(y) * (1 - sigmoid(y)) * self.train_data[i][j]
                elif self.act_function == 'relu':
                    if y > 0:
                        weight_gradient_accum[j] += 2 * (y - self.train_label[i]) * self.train_data[i][j]
                    else:
                        weight_gradient_accum[j] += 0
                else:
                    weight_gradient_accum[j] += 2 * (y - self.train_label[i]) * self.train_data[i][j]

            # Calculate bias gradients
            if self.act_function == 'sigmoid':
                bias_gradient_accum += 2 * (y - self.train_label[i]) * sigmoid(y) * (1 - sigmoid(y))
            elif self.act_function == 'relu':
                if y > 0:
                    bias_gradient_accum += 2 * (y - self.train_label[i])
                else:
                    bias_gradient_accum += 0
            else:
                bias_gradient_accum += 2 * (y - self.train_label[i])

            mse = (y - self.train_label[i]) ** 2
            mse_sum += mse

        # Keep track of average training error
        self.training_error[epoch] = mse_sum / train_data_length

        # Average weight and bias gradients
        average_weight_gradient = weight_gradient_accum / train_data_length
        average_bias_gradient = bias_gradient_accum / train_data_length

        # Update parameters
        for i in range(len(self.weights)):
            self.weights[i] -= average_weight_gradient[i] * self.learning_rate
        self.bias -= average_bias_gradient * self.learning_rate

        # Use updated parameters to check accuracy on validation data
        valid_data_length = len(self.valid_data)
        mse_sum = 0
        for i in range(valid_data_length):
            z = np.dot(self.valid_data[i], self.weights) + self.bias
            y = self.apply_activation_function(z)
            mse = (y - self.valid_label[i]) ** 2
            mse_sum += mse
            if (y > self.threshold and self.valid_label[i] == 1) or (y <= self.threshold and self.valid_label[i] == 0):
                validation_correct_count += 1

        # Keep track of average validation error
        self.validation_error[epoch] = mse_sum / valid_data_length

        # Calculate and keep track of training and validation accuracies
        self.training_accuracy[epoch] = training_correct_count / train_data_length
        self.validation_accuracy[epoch] = validation_correct_count / valid_data_length

    def apply_activation_function(self, z):
        if self.act_function == 'sigmoid':
            return sigmoid(z)
        elif self.act_function == 'relu':
            return max(0, z)
        else:
            return z

    def plot_data(self):
        plt.plot(self.training_error, color='black')
        plt.plot(self.validation_error, color='green')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.title('Average Training & Validation Losses (Sigmoid)')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.savefig('loss')
        plt.show()

        plt.plot(self.training_accuracy)
        plt.plot(self.validation_accuracy)
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.title('Training & Validation Accuracies (Sigmoid)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('accuracies')
        plt.show()

        print('Final weights:', self.weights)

        dispKernel(self.weights, 3, 9)


# Pseudocode
# Initialize parameters to random numbers
# Start running epochs
# Loop through all training data
# Calculate Z for current training datum
# Dot product with weights
# Add bias
# Run through activation function
# Compute gradient of loss function wrt each parameter (w_i, b)
# Add gradient to gradient accumulator for current epoch
# Average gradient values from accum
# Modify parameters to minimize loss
# Calculate training and validation accuracy (***)


# Instantiate and run neural network
SNC = SingleNeuronClassifier(args)
SNC.train()
SNC.plot_data()
