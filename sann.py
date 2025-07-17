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


def sum_inputs(inputs: list[tuple[float, float]]) -> float:
    """
    Calculate the activation value from a list of pairs of "x" input values
    and "w" weights. This is essentially just the dot product.
    """
    return sum([x * w for x, w in inputs])


def sigmoid(a: float, t: float = 0.0, r: float = 0.5) -> float:
    """
    Calculate the output value of a sigmoid based node.

    Take the activation value "a", a threshold value "t", and a shape parameter
    "r", and return the output value found somewhere on an s-shaped sigmoid
    curve.
    """
    return 1 / (1 + math.exp(-((a - t) / r)))


def tlu(a: int | float, t: int | float = 0) -> int:
    """
    Calculate the output value of a threshold logic unit (TLU).

    If the activation (a) is greater than the threshold (t), set the output to
    1 (truthy), else set it to 0 (falsey).
    """
    return 1 if a >= t else 0


def create_ann(structure: list) -> dict:
    """
    Return a dict representing a simple artificial neural network (ANN).

    The structure argument should be a list containing the number of nodes in
    each layer of a fully connected feed-forward neural network.

    The resulting dictionary will contain a list of layers, where each layer
    is a list of nodes. Each node is represented as a dictionary containing
    its incoming weights from the previous layer and a bias value. The weights
    and bias are randomly initialised to a value between -1 and 1.

    The first layer is ignored since it is the input layer and has no weights
    nor bias associated with it. There must be at least two layers (an input
    layer and an output layer) for the ANN to be valid.

    Other arbitrary arbitrary properties can be added to the returned
    dictionary, such as a fitness score, which can be used for training or
    evolution of the ANN, and a structure that defines the topology of the
    ANN (i.e. the number of nodes in each layer).
    """
    if len(structure) < 2:
        raise ValueError(
            "ANN must have at least two layers (input and output)."
        )
    layers = []
    # Create nodes with random weights and a bias for each layer except the
    # input layer
    for i in range(1, len(structure)):
        layer = []
        for j in range(structure[i]):
            layer.append(
                {
                    "weights": [
                        random.uniform(-1, 1) for _ in range(structure[i - 1])
                    ],
                    "bias": random.uniform(-1, 1),
                }
            )
        layers.append(layer)
    result = {"structure": structure, "fitness": None, "layers": layers}
    return result


def forward_pass(ann: dict, inputs: list) -> list:
    """
    Perform a forward pass through the ANN using the given inputs.

    The inputs are a list of values that are fed into the first layer of the
    ANN. The output of each layer is calculated and passed to the next layer
    until the final output is produced.
    """
    outputs = inputs
    for layer in ann["layers"]:
        new_outputs = []
        for node in layer:
            activation = sum_inputs(zip(outputs, node["weights"]))
            # Store the output in the node, used for backpropagation
            node["output"] = sigmoid(activation, node["bias"])
            new_outputs.append(node["output"])
        outputs = new_outputs
    return outputs


def clean_ann(ann: dict) -> dict:
    """
    Remove the outputs stored in nodes to clean up the ANN, so only the
    weights and biases remain.
    """
    for layer in ann["layers"]:
        for node in layer:
            if "output" in node:
                del node["output"]
    return ann


def backpropagate(
    ann: dict, inputs: list, expected_outputs: list, learning_rate: float = 0.1
) -> dict:
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
    for i in reversed(range(len(ann["layers"]))):
        layer = ann["layers"][i]

        # Get inputs to this layer
        if i == 0:
            layer_inputs = inputs
        else:
            layer_inputs = [node["output"] for node in ann["layers"][i - 1]]

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
            previous_layer = ann["layers"][i - 1]
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
    ann: dict,
    training_data: list[tuple[list[float], list[float]]],
    epochs: int = 1000,
    learning_rate: float = 0.1,
    log=lambda x: None,
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
    layers: list[int],
    population: int,
    generate: callable,
    fitness: callable,
    halt: callable,
    reverse: bool = True,
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
    returns a fitness score that should also be annotated as the network's
    ann["fitness"] value. The halt function takes the current population
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


def roulette_wheel_selection(population: list[dict]) -> dict:
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

    random_point = random.uniform(0.0, total_fitness)

    fitness_tally = 0.0
    for ann in population:
        if "fitness" in ann:
            fitness_tally += ann["fitness"]
        if fitness_tally > random_point:
            return ann


def crossover(mum: dict, dad: dict) -> tuple[dict, dict]:
    """
    Perform crossover between two parent ANNs (mum and dad) to create two
    child ANNs. The children inherit weights and biases from both parents
    through the following process:

    1. Two split points are chosen randomly. A split point is always at the
       boundary between two nodes in a layer.
    2. The first child inherits weights and biases from the mum up to the first
       split point, then from the dad until the second split point, and finally
       from the mum again.
    3. The second child inherits weights and biases from the dad up to the first
       split point, then from the mum until the second split point, and finally
       from the dad again.
    4. Nodes are treated as a continuous sequence across layers, so the split
       points can cross layer boundaries.
    5. The children are returned as a tuple of two new ANN structures.
    """
    # Flatten the nodes in both parents to treat them as a continuous sequence.
    # This makes it easier to choose split points across layers.
    flat_mum = [node for layer in mum["layers"] for node in layer]
    flat_dad = [node for layer in dad["layers"] for node in layer]

    # Choose two random split points, ensuring split1 < split2.
    split1 = random.randint(0, len(flat_mum) - 2)
    split2 = random.randint(split1 + 1, len(flat_mum) - 1)

    # Create children by slicing and combining parts from both parents.
    child1 = flat_mum[:split1] + flat_dad[split1:split2] + flat_mum[split2:]
    child2 = flat_dad[:split1] + flat_mum[split1:split2] + flat_dad[split2:]

    # Reshape flat children back into ANN expressed as layers.
    def reshape_to_layers(flat_ann, layers):
        reshaped = []
        index = 0
        for layer_size in layers:
            reshaped.append(flat_ann[index : index + layer_size])
            index += layer_size
        return reshaped

    child1 = {
        "layers": reshape_to_layers(child1, mum["structure"][1:]),
        "structure": mum["structure"],
        "fitness": None,
    }
    child2 = {
        "layers": reshape_to_layers(child2, dad["structure"][1:]),
        "structure": dad["structure"],
        "fitness": None,
    }
    return child1, child2
