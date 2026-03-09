# akshar

[(open in streamlit)](https://akshar-mohit-ambe.streamlit.app/)

## Hindi Character Recognition from Scratch

Most machine learning projects begin and end inside a framework. You define layers, train, and tune hyperparameters. I wanted to understand what happens behind the scenes.

So I built a convolutional neural network entirely from scratch in pure Python (and some C) to recognize handwritten Hindi/Devanagari consonants.

No PyTorch. No TensorFlow. Just tensors, gradients, and a lot of debugging.

---

## Project Inspiration and Dataset

I initially wanted to work with MNIST, the 'Hello World' of machine learning projects. After noticing the data is on the easier side to learn and solve, as well as being a common (even overused) dataset, I sought something more complex.

After searching a number of datasets for character datasets, I found decade old [research](https://arxiv.org/abs/2507.10398) on using a CNN which performed strong (99% training accuracy) on classifying the 36 consonants. The dataset contains 2000 samples of each character, which was then split 85%-15% for training and testing.

---

## Designing the Network

I used a straightforward convolutional architecture adapted from the paper:

* 2 Convolution layers to extract features
* ReLU activations for nonlinearity
* Pooling layers to downsample resolution
* A final dense layer to map features to class probabilities

Conceptually, the network works in stages:

### Stage 1: Local Stroke Detection

Early layers detect edges, curves, and small stroke fragments.

### Stage 2: Structural Composition

Mid-level features combine local strokes into larger shapes.

### Stage 3: Character Identity
The dense layer interprets the full feature map as a specific consonant.

---

## The Tensor System

Instead of relying on NumPy for everything, I implemented a custom tensor structure that can represent images (2D), convolutions with channels (3D), and even training batches (4D) and more complex data.

To optimize data retrieval for these high dimensional objects, the values are stored in one long list, which is accessed using a index system.

This forced me to reason about how data is actually stored in memory. Reshaping and flattening became deliberate operations rather than implicit conveniences. It also made debugging much harder, but the tradeoff of readability and faster computation was worth it.

---

## Implementing Convolution Manually

At its core, a convolution is:

* Sliding a kernel over the image
* Multiplying overlapping regions (like a dot product)
* Summing the results and arranging them as a new, smaller image

But making it work in a CNN also needs:

* activations for backpropagation
* gradients for the weights (and bias)
* updating these parameters

Writing backward passes manually was the most educational (and difficult) part of the project. Once implemented, I explored performance optimizations like reorganizing loops and precalculating values.

---

## Pooling

To reduce the complexity of the network, each convolution also receives pooling to reduce total parameters later on.

Pooling follows a similar algorithm to convolutions

* Considering pixels in a kernel sliding over the image
* Selecting features (the max or average of the window)
* Arranging the results as a new, even smaller image

Backpropagation acts the same:

* Pass gradients back to the selected features in each window
* No need to calculate new gradients / update any parameters

---

## Other Layers & Optimizations

After convolution and pooling, the image is flattened into one row of nodes, then passed through a dense layer, where every input is connected to all 36 of the class nodes.

For this functionality I made a number of classes that perform standard neural network behavior:

* Flatten
* Dense
* Activation
* Loss

Finally, all of the layers can be passed to a CNN class which trains and tests a model.

On the first pass, the network failed to finish the epoch in a reasonable amount of time..it would have taken almost *3 months* of running to finish :(

To solve this, I researched a number of optimizations I could add to improve time complexity (ex. im2col and fft for convolutions, gpu programming, multithreading), and ending up going with a simpler option: using C to chop memory overhead as well as reduce calculation runtime.

The following results were obtained while training on MNIST using 2 convolution cycles and 1 dense layer.

Compiling a C package to back the layers with trainable parameters or heavy looping allowed me to end with a training speed of about **0.01** seconds per sample:

    Approach                        Total Time             Sample Time

    Tensor (Raw Python):            DNF                    120.0s
    
    Hoisted Loops:                  3.3h                   0.200s
    
    C-backed Convolve               15m                    0.015s
    
    C-backed Dense                  13.3m                  0.013s
    
    C-backed Pooling                11m                    0.011s

---

## Achievements (and Improvements)

After training on labeled Devanagari consonants for 50 epochs, the network peaked at **94.12%** when testing.

An alternate model with an additional convolutional layer learned slightly better, with **95.27%** on test data.

I'm proud of this since it competes fairly well with the 2015 paper (96.36% testing), but I could definitely make some improvements:

* Data augmentation to predict images independently of position/rotation
* Regularization techniques
* Better preprocessing + optimization
* learning the entire Hindi alphabet (vowels and numerals)

---

## The End! + Resources

If you would like to build something like this, I would suggest reading/watching these:

- [Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
- [Devanagari Handwritten Character Recognition using Convolutional Neural Network | arXiv](https://arxiv.org/abs/2507.10398)
- [Convolutional Neural Networks from Scratch | In Depth](https://www.youtube.com/watch?v=jDe5BAsT2-Y)
- [Convolutional Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=Lakz2MoHy6o)
- [Code Sample 1](https://github.com/as791/Devanagari-character-recognition/blob/master/Devanagri_recognation.ipynb)
- [Code Sample 2](https://www.kaggle.com/code/roochathatte/roochaathatte-devcharrecog-nhcd-scdc)
