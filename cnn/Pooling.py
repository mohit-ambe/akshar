from cnn.Tensor import Tensor


class Pooling:
    def __init__(self, ph, pw, stride=None, mode="max"):
        self.ph = ph
        self.pw = pw
        self.stride = stride if stride is not None else ph
        if mode not in ("max", "avg"):
            raise ValueError("mode must be 'max' or 'avg'")
        self.mode = mode

        self.x = None
        self.argmax = None

    def zero_grad(self):
        pass

    def forward(self, x, train=True):
        C, H, W = x.shape
        ph, pw, s = self.ph, self.pw, self.stride

        Hout = (H - ph) // s + 1
        Wout = (W - pw) // s + 1

        y = Tensor.zeros((C, Hout, Wout))
        argmax = Tensor.zeros((C, Hout, Wout))

        xoff, yoff = x.offset, y.offset
        sx0, sx1, sx2 = x.strides
        sy0, sy1, sy2 = y.strides

        if self.mode == "max":
            amdata = argmax.data
            amoff = argmax.offset
            sam0, sam1, sam2 = argmax.strides

            for c in range(C):
                x_c = xoff + c * sx0
                y_c = yoff + c * sy0
                am_c = amoff + c * sam0

                for oy in range(Hout):
                    iy0 = oy * s
                    y_cy = y_c + oy * sy1
                    am_cy = am_c + oy * sam1

                    for ox in range(Wout):
                        ix0 = ox * s
                        out_idx = y_cy + ox * sy2

                        best_val = -float("inf")
                        best_k = 0
                        k = 0

                        x_win_row0 = x_c + iy0 * sx1
                        for ky in range(ph):
                            x_row = x_win_row0 + ky * sx1
                            x_col0 = x_row + ix0 * sx2
                            for kx in range(pw):
                                v = x.data[x_col0 + kx * sx2]
                                if v > best_val:
                                    best_val = v
                                    best_k = k
                                k += 1

                        y.data[out_idx] = best_val
                        amdata[am_cy + ox * sam2] = float(best_k)

        else:
            inv = 1.0 / (ph * pw)

            for c in range(C):
                x_c = xoff + c * sx0
                y_c = yoff + c * sy0

                for oy in range(Hout):
                    iy0 = oy * s
                    y_cy = y_c + oy * sy1

                    for ox in range(Wout):
                        ix0 = ox * s
                        out_idx = y_cy + ox * sy2

                        acc = 0.0
                        x_win_row0 = x_c + iy0 * sx1
                        for ky in range(ph):
                            x_row = x_win_row0 + ky * sx1
                            x_col0 = x_row + ix0 * sx2
                            for kx in range(pw):
                                acc += x.data[x_col0 + kx * sx2]

                        y.data[out_idx] = acc * inv

        if train:
            self.x = x
            self.argmax = argmax if self.mode == "max" else None

        return y

    def backward(self, dy):
        x = self.x
        C, H, W = x.shape
        C2, Hout, Wout = dy.shape

        ph, pw, s = self.ph, self.pw, self.stride
        dx = Tensor.zeros((C, H, W))

        dxoff, dyoff = dx.offset, dy.offset
        sdx0, sdx1, sdx2 = dx.strides
        sdy0, sdy1, sdy2 = dy.strides

        if self.mode == "max":
            am = self.argmax
            amdata = am.data
            amoff = am.offset
            sam0, sam1, sam2 = am.strides

            for c in range(C):
                dx_c = dxoff + c * sdx0
                dy_c = dyoff + c * sdy0
                am_c = amoff + c * sam0

                for oy in range(Hout):
                    dy_cy = dy_c + oy * sdy1
                    am_cy = am_c + oy * sam1
                    iy0 = oy * s

                    for ox in range(Wout):
                        g = dy.data[dy_cy + ox * sdy2]

                        k = int(amdata[am_cy + ox * sam2])
                        ky = k // pw
                        kx = k - ky * pw

                        iy = iy0 + ky
                        ix = ox * s + kx

                        dx_idx = dx_c + iy * sdx1 + ix * sdx2
                        dx.data[dx_idx] += g
        else:
            scale = 1.0 / (ph * pw)

            for c in range(C):
                dx_c = dxoff + c * sdx0
                dy_c = dyoff + c * sdy0

                for oy in range(Hout):
                    dy_cy = dy_c + oy * sdy1
                    iy0 = oy * s

                    for ox in range(Wout):
                        g = dy.data[dy_cy + ox * sdy2] * scale
                        ix0 = ox * s

                        dx_win_row0 = dx_c + iy0 * sdx1
                        for ky in range(ph):
                            dx_row = dx_win_row0 + ky * sdx1
                            dx_col0 = dx_row + ix0 * sdx2
                            for kx in range(pw):
                                dx.data[dx_col0 + kx * sdx2] += g

        return dx

    def step(self):
        pass

    def __str__(self):
        return f"Pooling({self.ph}, {self.pw}, stride={self.stride}, mode='{self.mode}')"