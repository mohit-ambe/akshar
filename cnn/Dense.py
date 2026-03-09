import math
from random import gauss, seed
from cnn.Tensor import Tensor


class Dense:
    def __init__(self, din, dout, softmax=False, lr=0.01, random_seed=1):
        self.din = din
        self.dout = dout
        self.softmax = softmax
        self.lr = lr
        self.random_seed = random_seed

        seed(self.random_seed)

        scale = math.sqrt(1.0 / din)

        self.W = Tensor([gauss(0.0, scale) for _ in range(din * dout)], (din, dout))
        self.b = Tensor.zeros((dout,))

        self.dW = Tensor.zeros((din, dout))
        self.db = Tensor.zeros((dout,))

        self.x = None
        self.z = None
        self.y = None

    def zero_grad(self):
        self.dW.zero()
        self.db.zero()

    @staticmethod
    def softmax(z):
        D = z.shape[0]
        y = Tensor.zeros((D,))

        zdata, ydata = z.data, y.data
        zoff, yoff = z.offset, y.offset
        sz0 = z.strides[0]
        sy0 = y.strides[0]

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
        z = Tensor.zeros((self.dout,))

        xdata, zdata, wdata, bdata = x.data, z.data, self.W.data, self.b.data
        xoff, zoff = x.offset, z.offset
        woff, boff = self.W.offset, self.b.offset

        sx0 = x.strides[0]
        sz0 = z.strides[0]
        sw0, sw1 = self.W.strides
        sb0 = self.b.strides[0]

        for j in range(self.dout):
            acc = bdata[boff + j * sb0]
            w_col = woff + j * sw1
            for k in range(self.din):
                acc += xdata[xoff + k * sx0] * wdata[w_col + k * sw0]
            zdata[zoff + j * sz0] = acc

        y = Dense.softmax(z) if self.softmax else z

        if train:
            self.x = x
            self.z = z
            self.y = y

        return y

    def backward(self, dy):
        dz = dy
        dx = Tensor.zeros((self.din,))

        x = self.x
        xdata, dzdata, dxdata = x.data, dz.data, dx.data
        wdata = self.W.data
        dWdata, dbdata = self.dW.data, self.db.data

        xoff, dzoff, dxoff = x.offset, dz.offset, dx.offset
        woff = self.W.offset
        dWoff, dboff = self.dW.offset, self.db.offset

        sx0 = x.strides[0]
        sdz0 = dz.strides[0]
        sdx0 = dx.strides[0]

        sw0, sw1 = self.W.strides
        sdw0, sdw1 = self.dW.strides
        sdb0 = self.db.strides[0]

        for j in range(self.dout):
            g = dzdata[dzoff + j * sdz0]
            dbdata[dboff + j * sdb0] += g

        for j in range(self.dout):
            g = dzdata[dzoff + j * sdz0]
            w_col = woff + j * sw1
            dW_col = dWoff + j * sdw1

            for k in range(self.din):
                xk = xdata[xoff + k * sx0]
                dWdata[dW_col + k * sdw0] += xk * g
                dxdata[dxoff + k * sdx0] += wdata[w_col + k * sw0] * g

        return dx

    def step(self):
        for j in range(self.dout):
            self.b[(j,)] = self.b.item((j,)) - self.lr * self.db.item((j,))

        for j in range(self.dout):
            for k in range(self.din):
                self.W[(k, j)] = self.W.item((k, j)) - self.lr * self.dW.item((k, j))

    def __str__(self):
        return f"Dense({self.din}, {self.dout}, {self.softmax}, {self.lr}, {self.random_seed})"