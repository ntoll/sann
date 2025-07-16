# Simple Artificial Neural Network (sann.py) 👶🤖🧠🎓

A naive Python implementation of an ANN that's useful for educational purposes
and clarifying the concepts of feed-forward neural networks, backpropagation,
neuro-evolution of weights and biases, and activation functions. This
implementation is not intended for production use or performance-critical
applications. 😉

See [Behind the AI Curtain](https://ntoll.org/article/ai-curtain/) for a 
comprehensive exploration of the concepts behind this code.

This module should work with both [CPython](https://python.org) and
[MicroPython](https://micropython.org/).

Try a couple of examples of this library in use online via PyScript:

* [Backpropagated numeral recognition](https://pyscript.com/@ntoll/sann-character-recognition/latest) - 
  a neural network that underwent supervised training will categorise hand
  written numerals from a corpus of unseen test data. ✍️⁉️
* [Neuro-evolved snake game]() - 
  the classic "SNAKE" game, but played by a neural network that underwent
  unsupervised neuro-evolution to arrive at a player best adapted to the 2D
  world of "SNAKE". It's a snAIke. 🤖🐍

## Installation 📦

If you're using CPython:

1. Create a virtual environment.
2. `pip install sann`

For MicroPython, just copy the `sann.py` file somewhere on your Python path.

## Usage 💪

**SANN is for educational use only.**

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

The network is expressed as a dictionary with three attributes:

1. `structure` - a list defining the number of nodes in each layer of the
   network (i.e. what was passed into the `create_network` function to create 
   it)
2. `fitness` - by default set to `None`, but used during neuro-evolution to
   indicate the arbitrary fitness score of the network during unsupervised
   training.
3. `layers` - a list of layers, with each layer containing a Python dict for 
   each node in the layer. Each dict contains a list of incoming weights and 
   a bias value, all of which are initialised with a random value between -1
   and 1. Since the input layer doesn't have associated weights nor bias
   (because its values are the raw input data), it is ignored.

```python
{
  "structure": [2, 3, 1],
  "fitness": None,
  "layers": [
  # No definition of the input layer needed, because its values are the raw 
  # input data
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
    ],
  ],
}
```

### Training

Training is the process through which the artificial neural network, created
with randomly generated weights and biases, is modified and refined so that 
it achieves some useful outcome. There are broadly two ways to do this:

* [Supervised training](https://en.wikipedia.org/wiki/Supervised_learning): 
  where the network is trained with labelled data. Put
  simply, given many examples of training input, the network is adjusted so it 
  produces the expected (labelled) output provided by humans. Once trained
  the network is tested with previously unseen labelled test data to check it
  correctly produces the expected results to the right level of accuracy. SANN
  provides [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) 
  capabilities as an example of this sort of training.
* [Unsupervised training](https://en.wikipedia.org/wiki/Unsupervised_learning):
  where the network is trained on data that is NOT labelled. This usually
  involves a training procedure that measures the accuracy or efficiency of
  the behaviour of the network in some way, combined with a process of 
  adjustment used to improve the network's outcomes. SANN provides a 
  [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) based 
  [neuro-evolution](https://en.wikipedia.org/wiki/Neuroevolution) capability 
  as an example of this sort of training.

SANN provides a means of achieving both types of training in the following
ways:

#### Supervised training (backpropagation)... ⁉️👍👎

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

#### Unsupervised training (neuro-evolution) 🐒🥕🪵

Alternatively, perhaps because supervised training is not possible due to the
context in which the neural network is used, unsupervised training via the 
evolution of a population of networks is required. 

This is achieved 

### Use the network 🛠️⚙️✅

Given a representation of a trained or evolved neural network and a list of
input values, use the `forward_pass` function to retrieve a list of output 
values caused by passing the inputs through the neural network.

That's it!

```python
```

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
