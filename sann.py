"""
Simple Artificial Neural Network (sann.py).

A naive Python implementation of an ANN that's useful for educational purposes
and clarifying the concepts of feed-forward neural networks, backpropagation,
neuro-evolution of weights and biases, and activation functions. This
implementation is not intended for production use or performance-critical
applications. ;-)

See: https://ntoll.org/article/ai-curtain/ for a comprehensive and informal
exploration of the concepts behind this code.

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
from functools import partial


def sum_inputs(inputs):
    """
    Calculate the activation value from a list of pairs of "x" input values
    and "w" weights. This is essentially just the dot product.
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
    return 1 if a >= t else 0


def create_ann(layers):
    """
    Return a list of nodes that represent a simple artificial neural network
    (ANN) from the given layer definition (a list containing the number of
    nodes in each layer of a fully connected feed-forward neural network).

    Each layer in the result is a list of nodes. Each node is a dictionary
    containing its incoming weights from the previous layer in a fully
    connected feed-forward network, and the node's bias value. These values
    are randomly initialised to a value between -1 and 1.

    The first layer is ignored since it is the input layer and has no weights
    nor bias associated with it. There must be at least two layers (an input
    layer and an output layer) for the ANN to be valid.
    """
    if len(layers) < 2:
        raise ValueError(
            "ANN must have at least two layers (input and output)."
        )
    ann = []
    # Create nodes with random weights and a bias for each layer except the
    # input layer
    for i in range(1, len(layers)):
        layer = []
        for j in range(layers[i]):
            layer.append(
                {
                    "weights": [
                        random.uniform(-1, 1) for _ in range(layers[i - 1])
                    ],
                    "bias": random.uniform(-1, 1),
                }
            )
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
        for node in layer:
            activation = sum_inputs(zip(outputs, node["weights"]))
            # Store the output in the node, used for backpropagation
            node["output"] = sigmoid(activation, node["bias"])
            new_outputs.append(node["output"])
        outputs = new_outputs
    return outputs


def clean_ann(ann):
    """
    Remove the outputs stored in nodes to clean up the ANN, so only the
    weights and biases remain.
    """
    for layer in ann:
        for node in layer:
            if "output" in node:
                del node["output"]
    return ann


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
    # Forward pass using existing function (stores outputs in nodes)
    final_outputs = forward_pass(ann, inputs)

    # Calculate initial errors for output layer
    output_errors = [
        expected - actual
        for expected, actual in zip(expected_outputs, final_outputs)
    ]

    # Backpropagate through all layers
    current_errors = output_errors
    for i in reversed(range(len(ann))):
        layer = ann[i]

        # Get inputs to this layer
        if i == 0:
            layer_inputs = inputs
        else:
            layer_inputs = [node["output"] for node in ann[i - 1]]

        # Update weights and biases for current layer
        for j, node in enumerate(layer):
            # Calculate gradient using this node's stored output
            gradient = (
                node["output"] * (1 - node["output"]) * current_errors[j]
            )

            # Update weights using inputs to this layer
            for k in range(len(node["weights"])):
                node["weights"][k] += (
                    learning_rate * gradient * layer_inputs[k]
                )

            # Update bias
            node["bias"] += learning_rate * gradient

        # Calculate errors for previous layer (if not input layer)
        if i > 0:
            new_errors = []
            previous_layer = ann[i - 1]
            for j in range(len(previous_layer)):
                error = sum(
                    node["output"]
                    * (1 - node["output"])
                    * current_errors[k]
                    * node["weights"][j]
                    for k, node in enumerate(layer)
                )
                new_errors.append(error)
            current_errors = new_errors

    return ann


def train(
    ann, training_data, epochs=1000, learning_rate=0.1, log=lambda x: None
):
    """
    Supervised training of the ANN using the provided training data.

    The training data is a list of tuples where each tuple contains inputs and
    the expected output. The ANN is trained for a specified number of epochs,
    adjusting the weights based on the error between actual and expected outputs.

    The log function can be used to log progress during training. It defaults
    to a no-op function that does nothing.
    """
    log("Training ANN...")
    for _ in range(epochs):
        log(f"Epoch {_ + 1}/{epochs}")
        for inputs, expected_outputs in training_data:
            backpropagate(ann, inputs, expected_outputs, learning_rate)
        log(clean_ann(ann))
    log("Training complete.")
    return ann


def evolve(
    layers,
    population,
    generate,
    fitness,
    halt,
    reverse=True,
    log=lambda x: None,
):
    """
    Evolve a population of ANNs using a genetic algorithm.

    The layers define the topology of the ANNs as a list of layer sizes (as
    per the create_ann function in this module). The population should be an
    integer defining the number of ANNs in each generation. The generate
    function takes the current population sorted by fitness and generates a
    new population for the next generation. The fitness function takes an
    individual ANN to evaluate and the current population (of siblings), and
    returns a fitness score that should also be annotated as the node's
    node["fitness"] value. The halt function takes the current population
    and generation count to determine if the genetic algorithm should stop.
    The reverse flag indicates if the fittest ANN has the highest (True) or
    lowest (False) fitness score. Finally, the log function can be used to
    log each generation during the course of evolution. It defaults to a
    no-op function that does nothing.

    When the genetic algorithm halts, it returns the final population
    ordered by fitness.
    """
    # Create initial population
    seed_generation = [create_ann(layers) for _ in range(population)]
    # Sort it by fitness
    current_population = sorted(
        seed_generation,
        key=partial(fitness, population=seed_generation),
        reverse=reverse,
    )
    generation_count = 0
    log(current_population)
    # Keep evolving until the halt function returns True.
    while not halt(current_population, generation_count):
        generation_count += 1
        new_generation = generate(current_population)
        current_population = sorted(
            new_generation,
            key=partial(fitness, population=new_generation),
            reverse=True,
        )
        log(current_population)
    return current_population


def roulette_wheel_selection(population):
    """
    A random number between 0 and the total fitness score of all the ANNs in
    a population is chosen (a point with a slice of a roulette wheel). The code
    iterates through the ANNs adding up the fitness scores. When the
    subtotal is greater than the randomly chosen point it returns the ANN
    at that point "on the wheel".

    See: https://en.wikipedia.org/wiki/Fitness_proportionate_selection
    """
    total_fitness = 0.0
    for ann in population:
        if "fitness" in ann:
            total_fitness += ann["fitness"]

    # Ensures random selection if no solutions are "fit".
    if total_fitness == 0.0:
        return random.choice(population)

    random_point = random.uniform(0.0, total_fitness)

    fitness_tally = 0.0
    for ann in population:
        if "fitness" in ann:
            fitness_tally += ann["fitness"]
        if fitness_tally > random_point:
            return ann
