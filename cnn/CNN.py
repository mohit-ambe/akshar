import time
from random import seed, shuffle

from cnn.Dense import Dense
from cnn.Tensor import Tensor


class CNN:
    def __init__(self, layers, loss_fn, random_seed=1):
        seed(random_seed)
        self.random_seed = random_seed

        self.layers = list(layers)
        self.loss_fn = loss_fn

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x

    def backward(self, dlogits):
        dy = dlogits
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def step(self):
        for layer in self.layers:
            layer.step()

    @staticmethod
    def _argmax_row(logits_row):
        K = logits_row.shape[0]
        best_j = 0
        best_v = logits_row.item((0,))
        for j in range(1, K):
            v = logits_row.item((j,))
            if v > best_v:
                best_v = v
                best_j = j
        return best_j

    @staticmethod
    def _accuracy(preds, labels):
        correct = 0
        for p, y in zip(preds, labels):
            if p == y:
                correct += 1
        return correct / max(1, len(labels))

    def train_epoch(self, x_train, y_train):
        N, H, W = x_train.shape
        if len(y_train) != N:
            raise ValueError("x_train first dim must equal len(y_train)")

        idxs = list(range(N))
        shuffle(idxs)

        total_loss = 0.0
        total_correct = 0

        # TIME = time.time()
        # t = 0
        for i in idxs:
            base = x_train.offset + i * (H * W)
            xb = Tensor(x_train.data[base: base + H * W], (1, H, W))
            yb = y_train[i]

            logits = self.forward(xb, train=True)

            loss = self.loss_fn.forward(logits, yb)
            dlogits = self.loss_fn.backward()

            self.zero_grad()
            self.backward(dlogits)
            self.step()

            # t += 1
            # if t % 100 == 0:
            #     print(t, " - ", round(time.time() - TIME, 4), "s",sep="")

            pred = self._argmax_row(logits)
            if pred == y_train[i]:
                total_correct += 1
            total_loss += loss

        # print(round(time.time() - TIME, 4), "s", sep="")
        return total_loss / max(1, N), total_correct / max(1, N)

    def eval_epoch(self, x_test, y_test):
        N, H, W = x_test.shape

        total_loss = 0.0
        total_correct = 0

        for i in range(N):
            base = x_test.offset + i * (H * W)
            xb = Tensor(x_test.data[base: base + H * W], (1, H, W))
            yb = y_test[i]

            logits = self.forward(xb, train=False)

            loss = self.loss_fn.forward(logits, yb)

            pred = self._argmax_row(logits)
            if pred == y_test[i]:
                total_correct += 1
            total_loss += loss

        return total_loss / max(1, N), total_correct / max(1, N)

    def predict(self, x):
        return list(Dense.softmax(self.forward(x, train=False)).data)

    @staticmethod
    def from_mw(filename, use_c=True):
        if use_c:
            from c_cnn.Convolve import Convolve
            from c_cnn.Dense import Dense
            from c_cnn.Pooling import Pooling
        else:
            from cnn.Convolve import Convolve
            from cnn.Dense import Dense
            from cnn.Pooling import Pooling

        from cnn.Activation import ReLU, Sigmoid
        from cnn.Flatten import Flatten
        from cnn.Loss import CrossEntropyLoss
        from cnn.Tensor import Tensor

        from array import array

        with open(filename, "r") as f:
            config = list(map(str.strip, f.readlines()))
        layers = []
        i = 0
        while i < len(config):
            line = config[i]
            if not line:
                i += 1
                continue
            if line.startswith("LOSS:"):
                break

            l = eval(line)
            if hasattr(l, "W") and hasattr(l, "b"):
                l.W = eval(config[i + 1])
                l.b = eval(config[i + 2])
                i += 3
            else:
                i += 1

            layers.append(l)

        loss = eval(line[len("LOSS:"):])

        return CNN(layers, loss)

    def to_mw(model, filename):
        out = []

        for layer in model.layers:
            out.append(layer.__str__())
            if hasattr(layer, "W") and hasattr(layer, "b"):
                out.append(layer.W.__str__())
                out.append(layer.b.__str__())
            out.append("")
        out.append("LOSS:" + model.loss_fn.__str__())

        try:
            open(filename, 'x')
        except FileExistsError:
            pass
        with open(filename, 'w') as f:
            for line in out:
                f.write(line + "\n")