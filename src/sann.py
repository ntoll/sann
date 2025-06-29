"""
Simple Artificial Neural Network (sann.py).

A naive Python implementation of an ANN that's useful for educational purposes
and clarifying the concepts of feed-forward neural networks, backpropagation,
and activation functions. This implementation is not intended for production use
or performance-critical applications. ;-)

Copyright (c) 2025 Nicholas H.Tollervey (ntoll@ntoll.org).

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import random


def sum_inputs(inputs):
    """
    Calculate the activation value from a list of pairs of "x" input values
    and "w" weights.
    """
    return sum([x * w for x, w in inputs])


def sigmoid(a, t=0, r=0.5):
    """
    Calculate the output value of a sigmoid based node.

    Take the activation value "a", a threshold value "t", and a shape parameter
    "r", and return the output value found somewhere on an s-shaped sigmoid
    curve.
    """
    return 1 / (1 + math.exp(-((a - t) / r)))


def tlu(a, t=0):
    """
    Calculate the output value of a threshold logic unit (TLU).

    If the activation (a) is greater than the threshold (t), set the output to
    1 (truthy), else set it to 0 (falsey).
    """
    return 1 if a > t else 0


def create_ann(layers):
    """
    Return a list of lists that represent a simple artificial neural network
    (ANN) from the given layer definition (a list containing the number of
    nodes in each layer of a fully connected feed-forward neural network).

    Each layer in the result is a list of weights for each node in a fully
    connected feed-forward network. The weights are randomly initialised to
    a value between -1 and 1.

    The first layer is ignored since it is the input layer and has no weights
    associated with its input. There must be at least two layers (an input
    layer and an output layer) for the ANN to be valid.
    """
    if len(layers) < 2:
        raise ValueError(
            "ANN must have at least two layers (input and output)."
        )
    ann = []
    for i in range(1, len(layers)):
        layer = []
        for j in range(layers[i]):
            weights = [random.uniform(-1, 1) for _ in range(layers[i - 1])]
            layer.append(weights)
        ann.append(layer)
    return ann


def forward_pass(ann, inputs):
    """
    Perform a forward pass through the ANN using the given inputs.

    The inputs are a list of values that are fed into the first layer of the
    ANN. The output of each layer is calculated and passed to the next layer
    until the final output is produced.
    """
    outputs = inputs
    for layer in ann:
        new_outputs = []
        for node_weights in layer:
            activation = sum_inputs(zip(outputs, node_weights))
            output = sigmoid(activation)
            new_outputs.append(output)
        outputs = new_outputs
    return outputs


def backpropagate(ann, inputs, expected_outputs, learning_rate=0.1):
    """
    Perform backpropagation to adjust the weights of the ANN based on the
    expected outputs.

    This function calculates the error for each node in the output layer,
    propagates that error back through the network, and adjusts the weights
    accordingly. The learning rate determines how much the weights are
    adjusted during each update.

    It returns the updated ANN with adjusted weights.
    """
    # Forward pass to get the actual outputs
    actual_outputs = forward_pass(ann, inputs)
    # Calculate output layer errors
    output_errors = [
        expected - actual
        for expected, actual in zip(expected_outputs, actual_outputs)
    ]
    # Backpropagate errors through the network
    for i in reversed(range(len(ann))):
        layer = ann[i]
        next_layer_errors = (
            output_errors if i == len(ann) - 1 else next_layer_errors
        )
        for j, node_weights in enumerate(layer):
            # Calculate the gradient for the current node
            gradient = (
                actual_outputs[j]
                * (1 - actual_outputs[j])
                * next_layer_errors[j]
            )
            # Update weights for the current node
            for k in range(len(node_weights)):
                node_weights[k] += (
                    learning_rate
                    * gradient
                    * (inputs[k] if i == 0 else ann[i - 1][k])
                )
        # Prepare next layer errors for backpropagation
        next_layer_errors = [
            sum(
                [
                    node_weights[k] * output_errors[j]
                    for j, node_weights in enumerate(layer)
                ]
            )
            for k in range(len(layer[0]))
        ]
    return ann
