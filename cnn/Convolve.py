import math
from random import gauss, seed
from cnn.Tensor import Tensor


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

        self.W = Tensor([gauss(0.0, scale) for _ in range(cout * cin * kh * kw)], (cout, cin, kh, kw))
        self.b = Tensor.zeros((cout,))

        self.dW = Tensor.zeros((cout, cin, kh, kw))
        self.db = Tensor.zeros((cout,))

        self.x = None

    def zero_grad(self):
        self.dW.zero()
        self.db.zero()

    def forward(self, x, train=True):
        Cin, H, W = x.shape
        Kh, Kw = self.kh, self.kw

        Hout = H - Kh + 1
        Wout = W - Kw + 1

        y = Tensor.zeros((self.cout, Hout, Wout))

        x_base = x.offset

        for oc in range(self.cout):
            b_oc = self.b.data[self.b.offset + oc * self.b.strides[0]]
            W_oc_base = self.W.offset + oc * self.W.strides[0]
            y_oc_base = y.offset + oc * y.strides[0]

            for oy in range(Hout):
                y_row_base = y_oc_base + oy * y.strides[1]
                iy0 = oy

                for ox in range(Wout):
                    acc = b_oc
                    ix0 = ox

                    for ic in range(Cin):
                        x_ic_base = x_base + ic * x.strides[0]
                        W_ocic_base = W_oc_base + ic * self.W.strides[1]

                        for ky in range(Kh):
                            x_row = x_ic_base + (iy0 + ky) * x.strides[1]
                            W_ocky = W_ocic_base + ky * self.W.strides[2]

                            for kx in range(Kw):
                                acc += (x.data[x_row + (ix0 + kx) * x.strides[2]] * self.W.data[
                                    W_ocky + kx * self.W.strides[3]])

                    y.data[y_row_base + ox * y.strides[2]] = acc

        if train:
            self.x = x

        return y

    def backward(self, dy):
        x = self.x
        Cin, H, W = x.shape
        Cout, Hout, Wout = dy.shape

        Kh, Kw = self.kh, self.kw

        dx = Tensor.zeros(x.shape)

        for oc in range(Cout):
            acc = 0.0
            dy_oc_base = dy.offset + oc * dy.strides[0]

            for oy in range(Hout):
                dy_row = dy_oc_base + oy * dy.strides[1]
                for ox in range(Wout):
                    acc += dy.data[dy_row + ox * dy.strides[2]]

            self.db.data[self.db.offset + oc * self.db.strides[0]] += acc

        for oc in range(Cout):
            dy_oc_base = dy.offset + oc * dy.strides[0]
            W_oc_base = self.W.offset + oc * self.W.strides[0]
            dW_oc_base = self.dW.offset + oc * self.dW.strides[0]

            for oy in range(Hout):
                dy_row = dy_oc_base + oy * dy.strides[1]
                iy0 = oy
                for ox in range(Wout):
                    g = dy.data[dy_row + ox * dy.strides[2]]
                    ix0 = ox

                    for ic in range(Cin):
                        x_ic_base = x.offset + ic * x.strides[0]
                        dx_ic_base = dx.offset + ic * dx.strides[0]
                        W_ocic_base = W_oc_base + ic * self.W.strides[1]
                        dW_ocic_base = dW_oc_base + ic * self.dW.strides[1]

                        for ky in range(Kh):
                            iy = iy0 + ky
                            x_row = x_ic_base + iy * x.strides[1]
                            dx_row = dx_ic_base + iy * dx.strides[1]
                            W_ocky = W_ocic_base + ky * self.W.strides[2]
                            dW_ocky = dW_ocic_base + ky * self.dW.strides[2]

                            for kx in range(Kw):
                                ix = ix0 + kx

                                xidx = x_row + ix * x.strides[2]
                                dxidx = dx_row + ix * dx.strides[2]
                                widx = W_ocky + kx * self.W.strides[3]
                                dwidx = dW_ocky + kx * self.dW.strides[3]

                                self.dW.data[dwidx] += x.data[xidx] * g
                                dx.data[dxidx] += self.W.data[widx] * g

        return dx

    def step(self):
        for oc in range(self.cout):
            bidx = self.b.offset + oc * self.b.strides[0]
            dbidx = self.db.offset + oc * self.db.strides[0]
            self.b.data[bidx] -= self.lr * self.db.data[dbidx]
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