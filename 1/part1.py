import numpy as np

# Load data
matrix = np.loadtxt("matrix.csv", delimiter=',')
vector = np.load("vector.npy")

output = []

for rowIndex in range(len(matrix)):
    accum = 0
    for colIndex in range(len(matrix[rowIndex])):
        accum += matrix[rowIndex][colIndex] * vector[colIndex][0]
    output.append([accum])

np.savetxt('output_forloop.csv', output)

output_2 = np.dot(matrix, vector)
np.save('output_dot.npy', output_2)

output_difference = output - output_2
np.savetxt('output_difference.csv', output_difference)
