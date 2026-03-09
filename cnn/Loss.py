import math

from cnn.Tensor import Tensor


class CrossEntropyLoss:
    def __init__(self, eps=1e-12):
        self.eps = eps

        self.y_true = None
        self.probs = None
        self.K = None

    def forward(self, pred, y_true):
        K = pred.shape[0]

        self.y_true = int(y_true)
        self.K = K

        m = -float("inf")
        for j in range(K):
            v = pred.item((j,))
            if v > m:
                m = v

        s = 0.0
        for j in range(K):
            s += math.exp(pred.item((j,)) - m)

        probs = Tensor.zeros((K,))
        p_y = 0.0
        for j in range(K):
            p = math.exp(pred.item((j,)) - m) / s
            probs[(j,)] = p
            if j == self.y_true:
                p_y = p

        self.probs = probs
        return -math.log(max(p_y, self.eps))

    def backward(self):
        K = self.K
        dp = Tensor.zeros((K,))

        for j in range(K):
            v = self.probs.item((j,))
            if j == self.y_true:
                v -= 1.0
            dp[(j,)] = v
        return dp

    def __str__(self):
        return "CrossEntropyLoss()"