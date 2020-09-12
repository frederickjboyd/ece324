# Used to load and plot data for section 5.6 in a3.pdf (Activation Function)

import numpy as np
import matplotlib.pyplot as plt

gradient_steps_relu = np.load('gradients_steps_relu.npy')
gradient_steps_sigmoid = np.load('gradients_steps_sigmoid.npy')
gradient_steps_tanh = np.load('gradients_steps_tanh.npy')

training_accuracies_relu = np.load('training_accuracies_relu.npy')
training_accuracies_sigmoid = np.load('training_accuracies_sigmoid.npy')
training_accuracies_tanh = np.load('training_accuracies_tanh.npy')

training_times_relu = np.load('training_times_relu.npy')
training_times_sigmoid = np.load('training_times_sigmoid.npy')
training_times_tanh = np.load('training_times_tanh.npy')

validation_accuracies_relu = np.load('validation_accuracies_relu.npy')
validation_accuracies_sigmoid = np.load('validation_accuracies_sigmoid.npy')
validation_accuracies_tanh = np.load('validation_accuracies_tanh.npy')

plt.plot(gradient_steps_relu, training_accuracies_relu)
plt.plot(gradient_steps_relu, validation_accuracies_relu)

plt.plot(gradient_steps_sigmoid, training_accuracies_sigmoid)
plt.plot(gradient_steps_sigmoid, validation_accuracies_sigmoid)

plt.plot(gradient_steps_tanh, training_accuracies_tanh)
plt.plot(gradient_steps_tanh, validation_accuracies_tanh)

plt.legend(['ReLU Training', 'ReLU Validation', 'Sigmoid Training', 'Sigmoid Validation', 'Tanh Training',
            'Tanh Validation'])

plt.title('Activation Function Comparison')
plt.xlabel('Gradient Step')
plt.ylabel('Accuracy')
plt.ylim(0.75, 0.85)
plt.savefig('fig', dpi=300)
# plt.show()

# Print time taken to train
print('ReLU Training Time:', training_times_relu[len(training_times_relu) - 1])
print('Sigmoid Training Time:', training_times_sigmoid[len(training_times_sigmoid) - 1])
print('Tanh Training Time:', training_times_tanh[len(training_times_tanh) - 1])
