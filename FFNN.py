import math

from numpy.random import normal
from numpy import dot, array, transpose
from numba import jit, vectorize


class FFNN:

    # Initialize the neural network
    def __init__(self, layer_nodes=(28, 2), learning_rate: float = 1):
        # Set amount of layers and its nodes by using a layer nodes parameter:
        # layer_nodes=(28, 2): in my case I had "28" input parameters and
        #                       was learning net to classify "2" subclasses of results
        # Example 1, to init the net with 1 hidden layer with 10 nodes: layer nodes = (28,10,2)
        # Example 2, to init the net with a 20 inputs, 3 hidden layers with 5 nodes in each: layer nodes = (20,5,5,5,2)
        self.layer_nodes = tuple(layer_nodes)

        # Learning rate
        self.lr = learning_rate

        # Link weight matrices, input_to_hidden1_weights and hidden1_to_output_weights etc.
        # Weights inside the arrays are w_i_j, where link is from node w to node w+1 in the next layer
        self.weights = [normal(0.0, pow(self.layer_nodes[w + 1], -0.5), (self.layer_nodes[w + 1], self.layer_nodes[w]))
                        for w in range(len(self.layer_nodes) - 1)]

    def train_net(self, inputs_list, targets_list):
        inputs = array(inputs_list, ndmin=2).T
        targets = array(targets_list, ndmin=2).T

        # Calculate signals into final output layer
        outputs = self.calculate_outputs(inputs)

        # The output layer error is the (target - outputs[-1]) and also is an entry parameter
        # Hidden layer error is the output layer error, split by weights, recombined at hidden nodes
        errors = self.calculate_errors([targets - outputs[-1]])

        # Update the weights for the links between the hidden and output layers
        outputs2 = outputs[::-1][1:]
        outputs2.append(inputs)
        for w, e, o1, o2 in zip(self.weights[::-1], errors, outputs[::-1], outputs2):
            w += self.configure_weights(self.lr, e, o1, o2)

    def ask_net(self, inputs_list):
        # Convert inputs list to 2d array
        inputs = array(inputs_list, ndmin=2).T
        return self.calculate_outputs(inputs)[-1]

    def calculate_errors(self, errors):
        for number, w in enumerate(self.weights[1:][::-1]):
            errors.append(dot(w.T, errors[number]))
        return errors

    def calculate_outputs(self, inputs):
        outputs = [inputs]
        # Calculate signals from the hidden layer to the final output layer
        for i in range(len(self.weights)):
            inputs_list = self.own_dot(self.weights[i], outputs[i])
            outputs.append(self.sigmoid(inputs_list))
        return outputs[1:]

    @staticmethod
    @jit(nopython=True, cache=True)
    def configure_weights(lr, errors, output, pre_outputs):
        """Function is decorated by @jit to make it work faster for a longer learning process"""
        return lr * dot((errors * output * (1.0 - output)), transpose(pre_outputs))

    @staticmethod
    @jit(nopython=True, cache=True)
    def own_dot(weight, output):
        """Function is decorated by @jit to make it work faster for a longer learning process"""
        return dot(weight, output)

    @staticmethod
    @vectorize(nopython=True, cache=True)
    def sigmoid(x):
        """An activation function is the sigmoid function.
           Function is decorated by @vectorize to make it work faster for a longer learning process"""
        return 1 / (1 + math.e ** (-x))
