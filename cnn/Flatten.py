from cnn.Tensor import Tensor


class Flatten:
    def __init__(self):
        self.orig_shape = None

    def zero_grad(self):
        pass

    def forward(self, x, train=True):
        if len(x.shape) != 3:
            raise ValueError("Flatten.forward expects x shape (C, H, W)")

        C, H, W = x.shape
        if train:
            self.orig_shape = x.shape

        return x.reshape((C * H * W,))

    def backward(self, dy):
        if self.orig_shape is None:
            raise RuntimeError("Must call forward(train=True) before backward")

        C, H, W = self.orig_shape
        if dy.shape != (C * H * W,):
            raise ValueError("Flatten.backward shape mismatch")

        return dy.reshape((C, H, W))

    def step(self):
        pass

    def __str__(self):
        return "Flatten()"