# Simple Artificial Neural Network (sann.py) 👶🤖🧠🎓

A naive Python implementation of an ANN that's useful for educational purposes
and clarifying the concepts of feed-forward neural networks, backpropagation,
and activation functions. This implementation is not intended for production use
or performance-critical applications. 😉

See [Behind the AI Curtain](https://ntoll.org/article/ai-curtain/) for a 
comprehensive exploration of the concepts behind this code.

This module should work with both [CPython](https://python.org) and
[MicroPython](https://micropython.org/).

## Installation 📦

If you're using CPython:

1. Create a virtual environment.
2. `pip install sann`

For MicroPython, just copy the `sann.py` file somewhere on your Python path.

## Usage 💪

**SANN** is for educational use only.

☠️☠️☠️ Do not use this code in production. ☠️☠️☠️

### Create a neural network ✨

The `create_network` function takes a list defining the number of nodes in each
layer. It returns a representation of a fully connected feed-forward neural
network.

This example creates a test network with three layers: an input layer with two
nodes, a hidden layer with three nodes, and an output layer with one node.

```python
import sann


my_nn = sann.create_network([2, 3, 1])
```

The network is expressed as a list of layers, with each layer containing a
Python dict for each node in the layer. Each dict contains a list of incoming 
weights and a bias value, all of which are initialised with a random value 
between -1 and 1.

Since the input layer doesn't have associated weights nor bias (because its
values are the raw input data), it is ignored.

```python
# No definition of the input layer needed, because its values are the raw 
# input data.
[
  [ # Hidden layer. Each hidden node has a bias and
    # two input weights: one each from the nodes in
    # the input layer.
    { # Node 1
      'bias': -0.08932407876323856,
      'weights': [
        0.9318837478301161,
        -0.3259188141579621
      ]
    },
    { # Node 2
      'bias': -0.7449380314648402,
      'weights': [
        -0.15786850474033964,
        -0.9455648956883143
      ]
    },
    { # Node 3
      'bias': 0.5168993191227431,
      'weights': [
        -0.8359684467197377,
        0.09538722516032427
      ]
    }
  ],
  [ # Output layer.
    { # A single node with a bias and input weights
      # from each of the three nodes in the hidden
      # layer.
      'bias': -0.46255520816058215,
      'weights': [
        0.991047585915775,
        -0.9995162202419827,
        0.15538558263904179
      ]
    }
  ]
]
```

### Train the network... ⁉️👍👎

For the network to be useful, it needs to be trained with example data. Such
data should be expressed as a list of pairs of values: the training input, and
the expected values in the output layer. Because the training data is annotated
with expected outcomes, this is called supervised training. Please see the
`examples/digit_recognition/train.py` file for an example of this process.

Use the `train` function to do exactly what it says:

```python
trained_nn = sann.train(
    my_nn, training_data, epochs=1000, learning_rate=0.1, log=print
)
```

The `train` function takes the initial randomly generated neural network, and
the `training_data` expressed as pairs of input/expected output, as described
above. The `epochs` value (default: 1000) defines the number of times the
training data is iterated over. The `learning_rate` (default: `0.1`) defines 
by how much weights and bias values are changed as errors are corrected. 
Finally, the optional `log` argument references a callable used to log
messages as the training process unfolds. It defaults to a no-op function with
no side-effect if it is not given.

The output of the `train` function is a representation of the network with the
refined weights and bias values.

Once trained, it is usual to check and evaluate the resulting neural network
with as-yet unseen test data. The `examples/digit_recognition/train.py` file
contains an example of this (see: `evaluate_model`). If the neural network is
not accurate enough in performance, perhaps consider adjusting the `epochs` or
`learning_rate` values, and re-train.

### ...or Evolve the network 🐒🥕🪵

Alternatively, perhaps because supervised training is not possible due to the
context in which the neural network is used, unsupervised training via the 
evolution of a population of networks is required. 

This is achieved 

### Use the network 🛠️⚙️✅

Given a representation of a trained or evolved neural network and some input 
values, use the `forward_pass` function to retrieve the output caused by
passing the input through the neural network.

That's it!

## Developer Setup 💻🧑‍💻

Before continuing, please read the statement about 
[care of community](./CARE_OF_COMMUNITY.md).

1. Clone the repository.
2. Create a virtual environment.
3. `pip install -r requirements.txt`
4. Run the test suite: `make check`
5. Educational examples in the `examples` directory.

See [Behind the AI Curtain](https://ntoll.org/article/ai-curtain/) for more
information.

## Feedback / Bugs 🗣️🐛❤️

Simply create an issue in the GitHub repository. If you're using this code
for something fun, please let me know.

Thank you! 🤗
