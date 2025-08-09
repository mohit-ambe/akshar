import io
import pandas as pd
from PIL import Image
from Matrix import Matrix
from NeuralNetwork import NeuralNetwork

splits = {'train':'mnist/train-00000-of-00001.parquet', 'test':'mnist/test-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])
print("Data loaded!")

def to_matrices(dataframe):
    images = []
    labels = []
    for _, row in dataframe.iterrows():
        image_bytes = io.BytesIO(row.iloc[0]['bytes'])
        image = Image.open(image_bytes)
        pixels = list(image.getdata())
        images.append(Matrix([pixels]).transpose() * (1 / 255))
        labels.append(Matrix(10, 1))
        labels[-1][row.iloc[1]][0] = 1
    return images, labels


train_images, train_labels = to_matrices(train_df.sample(30000))
test_images, test_labels = to_matrices(test_df.sample(5000))
print("Matrices created!")
del train_df
del test_df

model = NeuralNetwork(784, 10, layer_sizes=[128, 64, 32])
model.train(train_images, train_labels, 1)
acc, outputs = model.test(test_images, test_labels)

for i in range(len(test_labels)):
    idxmax = lambda nums:[i for i in range(len(nums)) if max(nums) == nums[i]][0]
    print(idxmax(test_labels[i].col(0)), idxmax(outputs[i].col(0)))

print(f"Test Accuracy: {acc:.2f}%")

while inp := input():
    print(eval(inp))

