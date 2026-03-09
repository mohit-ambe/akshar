from cnn.Tensor import Tensor
from c_cnn.c_extension import cnn


class Pooling:
    def __init__(self, ph, pw, stride=None, mode="max"):
        self.ph = ph
        self.pw = pw
        self.stride = stride if stride is not None else ph
        if mode not in ("max", "avg"):
            raise ValueError("mode must be 'max' or 'avg'")
        self.mode = mode

        # cache
        self.x = None  # (C, H, W)
        self.argmax = None  # (C, Hout, Wout) for max-pool

    def zero_grad(self):
        pass

    def forward(self, x, train=True):
        """x: Tensor (C, H, W) -> y: Tensor (C, Hout, Wout)"""
        if len(x.shape) != 3:
            raise ValueError("Pooling.forward expects x shape (C, H, W)")

        C, H, W = x.shape
        ph, pw, s = self.ph, self.pw, self.stride

        Hout = (H - ph) // s + 1
        Wout = (W - pw) // s + 1
        if Hout <= 0 or Wout <= 0:
            raise ValueError("Invalid output shape for pooling")

        if self.mode == "max":
            y_arr, argmax_arr = cnn.pool_forward_max(*x.args(), ph, pw, s)
            self.argmax = Tensor(argmax_arr, (C, Hout, Wout))
        else:
            y_arr = cnn.pool_forward_avg(*x.args(), ph, pw, s)

        if train:
            self.x = x

        y = Tensor(y_arr, (C, Hout, Wout))

        return y

    def backward(self, dy):
        """dy: Tensor (C, Hout, Wout) -> dx: Tensor (C, H, W)"""
        if self.x is None:
            raise RuntimeError("Must call forward(train=True) before backward")

        x = self.x
        C, H, W = x.shape
        if len(dy.shape) != 3:
            raise ValueError("Pooling.backward expects dy shape (C, Hout, Wout)")
        C2, Hout, Wout = dy.shape
        if C2 != C:
            raise ValueError(f"Channel mismatch: dy has C={C2}, expected {C}")

        ph, pw, s = self.ph, self.pw, self.stride

        if self.mode == "max":
            if self.argmax is None:
                raise RuntimeError("Missing argmax cache (did you call forward(train=True)?)")

            dx_arr = cnn.pool_backward_max(*dy.args(), *self.argmax.args(), H, W, ph, pw, s)
        else:
            dx_arr = cnn.pool_backward_avg(*dy.args(), H, W, ph, pw, s)

        dx = Tensor(dx_arr, (C, H, W))

        return dx

    def step(self):
        pass

    def __str__(self):
        return f"Pooling({self.ph}, {self.pw}, stride={self.stride}, mode='{self.mode}')"