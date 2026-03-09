from array import array


class Tensor:
    def __init__(self, data, shape, offset=0):
        self.data = Tensor._unpack(data)
        self.shape = tuple(shape)
        self.offset = int(offset)
        self.strides = Tensor._compute_strides(self.shape)

    @staticmethod
    def _unpack(collection):
        if isinstance(collection, array) and collection.typecode == 'd':
            return collection

        if all([type(c) != list for c in collection]):
            return array('d', map(float, collection))

        out = array('d')
        stack = [collection]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                for v in reversed(cur):
                    stack.append(v)
            else:
                out.append(float(cur))
        return out

    @staticmethod
    def _compute_strides(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def zeros(shape):
        n = 1
        for s in shape:
            n *= s
        return Tensor(array('d', [0.0]) * n, shape)

    def args(self):
        return memoryview(self.data), self.offset, *self.shape, *self.strides

    def zero(self):
        n = self.volume()
        o = self.offset
        self.data[o:o + n] = array('d', [0.0]) * n

    def volume(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def item(self, idx):
        i = self.offset + sum([self.strides[i] * idx[i] for i in range(len(idx))])
        return self.data[i]

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            raise TypeError("use tuple of slices")

        if len(slices) != len(self.shape):
            raise IndexError("rank mismatch")

        new_shape = []
        new_offset = self.offset

        for i, s in enumerate(slices):
            if not isinstance(s, slice):
                raise TypeError("only slices allowed in []")

            start = 0 if s.start is None else s.start
            stop = self.shape[i] if s.stop is None else s.stop
            step = 1 if s.step is None else s.step

            if step != 1:
                raise NotImplementedError("step != 1 not supported")

            if start < 0 or stop > self.shape[i] or start >= stop:
                raise IndexError("invalid slice")

            new_offset += start * self.strides[i]
            new_shape.append(stop - start)

        return Tensor(self.data, tuple(new_shape), offset=new_offset)

    def __setitem__(self, idx, value):
        idx = self.offset + sum([self.strides[i] * idx[i] for i in range(len(idx))])
        self.data[idx] = value

    def reshape(self, new_shape):
        n = 1
        for s in new_shape:
            n *= s
        if n != self.volume():
            raise ValueError("volume mismatch")
        return Tensor(self.data, new_shape, offset=self.offset)

    def __str__(self):
        return f"Tensor({self.data},{self.shape},{self.offset})"