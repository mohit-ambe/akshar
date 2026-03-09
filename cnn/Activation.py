import math

from cnn.Tensor import Tensor


class ReLU:
    def __init__(self):
        self.x = None

    def zero_grad(self):
        pass

    def forward(self, x, train=True):
        y = Tensor.zeros(x.shape)

        n = x.volume()
        x0 = x.offset
        y0 = y.offset

        for k in range(n):
            v = x.data[x0 + k]
            y.data[y0 + k] = v if v > 0.0 else 0.0

        if train:
            self.x = x
        return y

    def backward(self, dy):
        x = self.x

        dx = Tensor.zeros(x.shape)

        n = x.volume()
        x0 = x.offset
        dy0 = dy.offset
        dx0 = dx.offset

        for k in range(n):
            dx.data[dx0 + k] = dy.data[dy0 + k] * (1.0 if x.data[x0 + k] > 0.0 else 0.0)

        return dx

    def step(self):
        pass

    def __str__(self):
        return "ReLU()"


class Sigmoid:
    def __init__(self):
        self.y = None

    def zero_grad(self):
        pass

    def forward(self, x, train=True):
        y = Tensor.zeros(x.shape)

        n = x.volume()
        x0 = x.offset
        y0 = y.offset

        for k in range(n):
            v = x.data[x0 + k]
            y.data[y0 + k] = 1.0 / (1.0 + math.exp(-v))

        if train:
            self.y = y
        return y

    def backward(self, dy):
        y = self.y

        dx = Tensor.zeros(y.shape)

        n = y.volume()
        dy0 = dy.offset
        y0 = y.offset
        dx0 = dx.offset

        for k in range(n):
            s = y.data[y0 + k]
            dx.data[dx0 + k] = dy.data[dy0 + k] * s * (1.0 - s)

        return dx

    def step(self):
        pass

    def __str__(self):
        return "Sigmoid()"