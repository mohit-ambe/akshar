import math
from random import gauss, seed
from cnn.Tensor import Tensor
from c_cnn.c_extension import cnn

class Convolve:
    def __init__(self, cin, cout, kh, kw, lr=0.01, random_seed=1):
        self.cin = cin
        self.cout = cout
        self.kh = kh
        self.kw = kw
        self.lr = lr
        self.random_seed = random_seed

        seed(self.random_seed)

        scale = math.sqrt(2.0 / (cin * kh * kw))

        # Params
        self.W = Tensor([gauss(0.0, scale) for _ in range(cout * cin * kh * kw)], (cout, cin, kh, kw))
        self.b = Tensor.zeros((cout,))

        # Grads
        self.dW = Tensor.zeros((cout, cin, kh, kw))
        self.db = Tensor.zeros((cout,))

        # Cache
        self.x = None  # original input

    def zero_grad(self):
        # self.dW = Tensor.zeros((self.cout, self.cin, self.kh, self.kw))
        # self.db = Tensor.zeros((self.cout,))
        self.dW.zero()
        self.db.zero()

    def forward(self, x, train=True):
        """
        x: Tensor (Cin, H, W)
        returns: Tensor (Cout, Hout, Wout)
        """
        if len(x.shape) != 3:
            raise ValueError("Convolve.forward expects x shape (Cin, H, W)")

        Cin, H, W = x.shape
        if Cin != self.cin:
            raise ValueError(f"Cin mismatch: got {Cin}, expected {self.cin}")

        Kh, Kw = self.kh, self.kw

        Hout = H - Kh + 1
        Wout = W - Kw + 1
        if Hout <= 0 or Wout <= 0:
            raise ValueError("Invalid output shape")

        y_data = cnn.convolve_forward(*x.args(), *self.W.args(), *self.b.args(), self.cout, self.kh, self.kw)

        if train:
            self.x = x

        y = Tensor(y_data, (self.cout, Hout, Wout))

        return y

    def backward(self, dy):
        """
        dy: Tensor (Cout, Hout, Wout)
        returns dx: Tensor (Cin, H, W)
        """
        if self.x is None:
            raise RuntimeError("Must call forward(train=True) before backward")

        x = self.x
        Cin, H, W = x.shape
        Cout, Hout, Wout = dy.shape

        if Cout != self.cout:
            raise ValueError(f"Cout mismatch: got {Cout}, expected {self.cout}")

        dx_data = cnn.convolve_backward(*x.args(), *dy.args(), *self.W.args(), *self.dW.args(), *self.db.args(),
                                        self.cout, self.kh, self.kw)
        dx = Tensor(dx_data, x.shape)

        return dx

    def step(self):
        """SGD update (online)."""
        for oc in range(self.cout):
            # bias update
            bidx = self.b.offset + oc * self.b.strides[0]
            dbidx = self.db.offset + oc * self.db.strides[0]
            self.b.data[bidx] -= self.lr * self.db.data[dbidx]

            # weight update
            W_oc_base = self.W.offset + oc * self.W.strides[0]
            dW_oc_base = self.dW.offset + oc * self.dW.strides[0]

            for ic in range(self.cin):
                W_ocic = W_oc_base + ic * self.W.strides[1]
                dW_ocic = dW_oc_base + ic * self.dW.strides[1]

                for ky in range(self.kh):
                    W_ocky = W_ocic + ky * self.W.strides[2]
                    dW_ocky = dW_ocic + ky * self.dW.strides[2]

                    for kx in range(self.kw):
                        idx = kx * self.W.strides[3]
                        self.W.data[W_ocky + idx] -= self.lr * self.dW.data[dW_ocky + idx]

    def __str__(self):
        return f"Convolve({self.cin}, {self.cout}, {self.kh}, {self.kw}, {self.lr}, {self.random_seed})"