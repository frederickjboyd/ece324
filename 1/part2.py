import numpy as np


class ElementwiseMultiply(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, number):
        return np.multiply(self.weight, number)


class AddBias(object):
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, number):
        return self.bias + number


class LeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, number):
        return np.where(number < 0, self.alpha * number, number)


class Compose(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, numbers):
        for computation in self.layers:
            numbers = computation(numbers)
        return numbers


# # Testing
# # ElementwiseMultiply
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])
#
# instance = ElementwiseMultiply(array1)
# print(instance(array2))
#
# newInstance = ElementwiseMultiply(3)
# print(newInstance(5))
#
# # AddBias
# biasTest = AddBias(3)
# print(biasTest(np.array([10, 20, 30, 40, 100])))
# print(biasTest(6))
#
# # LeakyRelu
# reluTest = LeakyRelu(0.5)
# print(reluTest(np.array([-6, -4, -2, 0, 2, 4, 6])))
