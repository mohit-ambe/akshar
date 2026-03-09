import math
from random import gauss, seed
from cnn.Tensor import Tensor
from c_cnn.c_extension import cnn


class Dense:
    """
    Online-only Dense layer (no batch dimension).

    Shapes:
      x:  (Din,)
      z/y:(Dout,)
    """

    def __init__(self, din, dout, softmax=False, lr=0.01, random_seed=1):
        self.din = din
        self.dout = dout
        self.softmax = softmax
        self.lr = lr
        self.random_seed = random_seed

        seed(self.random_seed)

        scale = math.sqrt(1.0 / din)

        # Parameters
        self.W = Tensor([gauss(0.0, scale) for _ in range(din * dout)], (din, dout))
        self.b = Tensor.zeros((dout,))

        # Gradients
        self.dW = Tensor.zeros((din, dout))
        self.db = Tensor.zeros((dout,))

        # caches
        self.x = None
        self.z = None
        self.y = None  # only meaningful if softmax=True

    def zero_grad(self):
        # self.dW = Tensor.zeros((self.din, self.dout))
        # self.db = Tensor.zeros((self.dout,))
        self.dW.zero()
        self.db.zero()

    @staticmethod
    def softmax(z):
        """
        z: Tensor (D,)
        returns: Tensor (D,)
        """
        if len(z.shape) != 1:
            raise ValueError("Dense.softmax expects z shape (D,)")

        D = z.shape[0]
        y = Tensor.zeros((D,))

        zdata, ydata = z.data, y.data
        zoff, yoff = z.offset, y.offset
        sz0 = z.strides[0]
        sy0 = y.strides[0]

        # numerical stability: subtract max
        m = -float("inf")
        for j in range(D):
            v = zdata[zoff + j * sz0]
            if v > m:
                m = v

        s = 0.0
        for j in range(D):
            s += math.exp(zdata[zoff + j * sz0] - m)

        for j in range(D):
            ydata[yoff + j * sy0] = math.exp(zdata[zoff + j * sz0] - m) / s

        return y

    def forward(self, x, train=True):
        """
        x: Tensor (Din,)
        returns: Tensor (Dout,)
        """
        if len(x.shape) != 1:
            raise ValueError("Dense.forward expects x shape (Din,)")
        if x.shape[0] != self.din:
            raise ValueError("Dense.forward input size mismatch")

        z_data = cnn.dense_forward(*x.args(), *self.W.args(), *self.b.args(), self.din, self.dout)
        z = Tensor(z_data, (self.dout,))

        y = Dense.softmax(z) if self.softmax else z

        if train:
            self.x = x
            self.z = z
            self.y = y

        return y

    def backward(self, dy):
        """
        dy:
          - if softmax=False: dL/dz (shape (Dout,))
          - if softmax=True:  dL/dy (assumed to already include softmax derivative, e.g. softmax-onehot)
        returns:
          dx: dL/dx (shape (Din,))
        """
        if self.x is None:
            raise RuntimeError("Must call forward(train=True) before backward")
        if len(dy.shape) != 1 or dy.shape[0] != self.dout:
            raise ValueError("Dense.backward expects dy shape (Dout,)")

        dx_data = cnn.dense_backward(*self.x.args(), *dy.args(), *self.W.args(), *self.dW.args(), *self.db.args(),
                                     self.din, self.dout)
        dx = Tensor(dx_data, (self.din,))

        return dx

    def step(self):
        # b -= lr * db
        for j in range(self.dout):
            self.b[(j,)] = self.b.item((j,)) - self.lr * self.db.item((j,))

        # W -= lr * dW
        for j in range(self.dout):
            for k in range(self.din):
                self.W[(k, j)] = self.W.item((k, j)) - self.lr * self.dW.item((k, j))

    def __str__(self):
        return f"Dense({self.din}, {self.dout}, {self.softmax}, {self.lr}, {self.random_seed})"