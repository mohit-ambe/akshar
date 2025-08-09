# the following libraries are allowed since they are builtin to Python
from math import log, e
from random import uniform, seed

from Matrix import Matrix


class Loss:
    """
    Loss Functions score outputs,
    allowing a neural network to 'learn' from training.
    """

    @staticmethod
    def sse(exp: Matrix, obs: Matrix, derivative=False):
        error = exp - obs
        if not derivative:
            # SSE = SUM { (y - y^) ** 2 }
            return Matrix().dot(error.col(0), error.col(0))
        else:
            # dSSE / dy^ = -2 * (y - y^)
            return error * -2

    @staticmethod
    def binary_cross_entropy(exp: Matrix, obs: Matrix, derivative=False):
        # bias values to prevent math error on log
        un_zero = lambda z:[_ + 1e-10 for _ in z]
        one_minus = lambda z:un_zero([abs(1 - _) for _ in z])
        exp, obs = un_zero(exp.col(0)), un_zero(obs.col(0))

        if not derivative:
            y_hat = list(map(log, obs))
            one_minus_y_hat = list(map(log, one_minus(obs)))

            # BCE = SUM { - (yi)*log(y^i) - (1-yi)*log(1-y^i) }
            res = -Matrix().dot(exp, y_hat) - Matrix().dot(one_minus(exp), one_minus_y_hat)
            return res / len(exp)
        else:
            # dBCE / dy^ = - y / y^ + (1-y) / (1-y^)
            res = [-y / yh + (1 - y) / (1 - yh) for y, yh in zip(exp, obs)]
            return Matrix([[r] for r in res])


class Activation:
    """
    Activation Functions map inputs to
    simulate the firing of a neuron.
    """

    @staticmethod
    def sigmoid(x: float, derivative=False):
        try:
            s = 1 / (1 + e ** -x)
        except OverflowError:
            s = 1 / (1 + 0)
        if not derivative:
            return s
        else:
            return s * (1 - s)

    @staticmethod
    def relu(x: float, derivative=False):
        if not derivative:
            return max(0, x)
        else:
            return 0 if x <= 0 else 1


class NeuralNetwork:
    """
    A Feedforward neural network.

    Parameters include loss and activation functions.

    Customize by layer structure, learn rate, and random seed.
    """

    def __init__(self, inputs: int,
                 outputs: int,
                 layer_sizes: list[int],
                 loss=Loss.sse,
                 activation=Activation.sigmoid,
                 learn_rate=0.01,
                 random_seed=1):

        self.weights = []
        self.biases = []
        self.loss = loss
        self.activation = activation
        self.learn_rate = learn_rate
        self.random_seed = random_seed
        seed(self.random_seed)

        layer_sizes = [inputs] + layer_sizes + [outputs]
        for i in range(len(layer_sizes) - 1):
            w = Matrix(layer_sizes[i + 1], layer_sizes[i])
            w = w.apply(lambda _:uniform(-0.5, 0.5))
            self.weights.append(w)

            b = Matrix(layer_sizes[i + 1], 1)
            self.biases.append(b)

    def forward_propagate(self, datapoint: Matrix):
        """
        Pass a datapoint through the network.

        Return its layer nodes,
        both before and after activation.
        """

        data = []
        for d in datapoint[:]:
            data += [[x] for x in d]
        datapoint = Matrix(data)

        activations = [datapoint]
        responses = [datapoint]

        for i in range(len(self.weights)):
            datapoint = self.weights[i] * datapoint + self.biases[i]
            responses.append(datapoint)
            datapoint = datapoint.apply(self.activation)
            activations.append(datapoint)

        return activations, responses

    def backward_propagate(self, activations, responses, label):
        """
        Calculates partials based on cost.

        Adjust weights and biases to minimize loss.
        """

        dL_da = self.loss(label, activations[-1], True)
        for i in range(len(self.weights), 0, -1):
            # How is loss affected by neurons values?
            da_dz = responses[i].apply(lambda x:self.activation(x, True))
            dL_dz = Matrix([[la * az] for la, az in zip(dL_da.col(0), da_dz.col(0))])

            # update the loss gradient w.r.t activation
            dz_da = self.weights[i - 1]
            dL_da = dz_da.transpose() * dL_dz

            # How is loss affected by weights and biases?
            dz_dw = activations[i - 1]
            dz_db = 1
            dL_dw = dL_dz * dz_dw.transpose()
            dL_db = dL_dz * dz_db

            # update the weights and biases to minimize the loss
            self.weights[i - 1] -= dL_dw * self.learn_rate
            self.biases[i - 1] -= dL_db * self.learn_rate

    def train(self, data: list[Matrix], labels: list[Matrix], epochs: int):
        """
        Run a full training cycle.
        """

        for epoch in range(1, epochs + 1):
            for datapoint, label in zip(data, labels):
                activations, responses = self.forward_propagate(datapoint)
                self.backward_propagate(activations, responses, label)

    def test(self, data: list[Matrix], labels: list[Matrix], loss_threshold=0.1):
        """
        Test the loss for each data point against a threshold.

        Return percent accuracy and generated outputs.
        """

        outputs = []
        acc = 0
        for datapoint, label in zip(data, labels):
            activations, responses = self.forward_propagate(datapoint)
            outputs.append(activations[-1])
            acc += 1 if self.loss(label, outputs[-1]) < loss_threshold else 0

        return acc / len(labels), outputs
