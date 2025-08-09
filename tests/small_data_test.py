import time
from random import seed, random

from Matrix import Matrix
from NeuralNetwork import NeuralNetwork

seed(0)


def generate_dataset(n_points):
    data = []
    labels = []
    for _ in range(n_points):
        x1 = random() * 2
        x2 = random() * 2
        label = 1 if x1 + x2 > 2 else 0  # classification rule
        data.append(Matrix([[x1], [x2]]))
        labels.append(Matrix([[label]]))
    return data, labels


train_data, train_labels = generate_dataset(3240)
test_data, test_labels = generate_dataset(65)

nn = NeuralNetwork(inputs=2, outputs=1, layer_sizes=[3, 2], learn_rate=.025)
epochs = 10
for _ in range(1, 21):
    x = time.time()
    nn.train(train_data, train_labels, epochs=epochs)
    t = time.time() - x
    acc, outputs = nn.test(test_data, test_labels, loss_threshold=0.05)
    print(f"Completed {_ * epochs} epochs, {acc * 100:.2f}% - {round(t / epochs, 2)}s/epoch")
