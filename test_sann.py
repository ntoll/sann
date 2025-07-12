"""
Test file for the sann package, using PyTest.
"""

import sann
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_ann():
    """
    Fixture to create a sample artificial neural network (ANN) for testing.
    This ANN has 2 layers: an input layer with 3 nodes and an output layer
    with 2 nodes.
    """
    return sann.create_ann([3, 2])


def test_tlu():
    """
    Test the threshold logic unit (TLU) activation function.
    The TLU should return 1 if the input is greater than 0.5,
    otherwise it should return 0.
    """
    assert sann.tlu(0.5, t=0.5) == 1
    assert sann.tlu(0.4, t=0.5) == 0
    assert sann.tlu(0.6, t=0.5) == 1


def test_create_ann(sample_ann):
    """
    Test the creation of an ANN with 2 layers.
    The first layer should have 3 nodes and the second layer should have 2
    nodes.
    """
    assert len(sample_ann) == 1  # One layer after input layer
    assert len(sample_ann[0]) == 2  # Two nodes in the output layer
    assert (
        len(sample_ann[0][0]["weights"]) == 3
    )  # A node has weights for 3 inputs
    assert "bias" in sample_ann[0][0]  # A node should have a bias


def test_create_ann_too_few_layers():
    """
    Test the creation of an ANN with too few layers.
    The function should raise a ValueError if the number of layers is less than 2.
    """
    with pytest.raises(ValueError):
        sann.create_ann([3])  # Only one layer provided


def test_forward_pass(sample_ann):
    """
    Test the forward pass through the ANN with sample inputs.
    The inputs should be a list of 3 values corresponding to the input
    layer.
    """
    inputs = [0.5, 0.2, 0.8]
    outputs = sann.forward_pass(sample_ann, inputs)

    # Check that the output is a list of length equal to the number
    # of nodes in the output layer
    assert len(outputs) == 2
    assert all(
        0 <= output <= 1 for output in outputs
    )  # Outputs should be between 0 and 1


def test_clean_ann(sample_ann):
    """
    Test the cleaning of the ANN by removing outputs from the nodes.
    """
    inputs = [0.5, 0.2, 0.8]
    # Perform a forward pass to populate outputs
    sann.forward_pass(sample_ann, inputs)
    # Check that outputs are present before cleaning
    assert "output" in sample_ann[0][0]
    # Clean the ANN to remove outputs
    cleaned_ann = sann.clean_ann(sample_ann)
    assert "output" not in cleaned_ann[0][0]


def test_backpropagate():
    """
    Test the backpropagation of errors through the ANN.
    This will check if the weights are updated correctly after a forward pass
    and backpropagation.
    """
    # Create a sample ANN with a hidden layer and an output layer.
    sample_ann = sann.create_ann([3, 5, 2])
    # Set the weights and biases to known values for testing.
    sample_ann[0][0]["weights"] = [0.5, 0.2, 0.8]
    sample_ann[0][0]["bias"] = 0
    # Perform a forward pass with sample inputs.
    inputs = [0.5, 0.2, 0.8]
    expected_outputs = [1, 0]  # Expected output for the test.
    outputs = sann.forward_pass(sample_ann, inputs)

    # Perform backpropagation.
    sann.backpropagate(sample_ann, inputs, expected_outputs)

    # Check if weights have been updated (not equal to initial state).
    assert sample_ann[0][0]["weights"] != [0.5, 0.2, 0.8]  # Example check.
    assert sample_ann[0][0]["bias"] != 0  # Bias should also be updated.

    # Check if the outputs after backpropagation are different.
    new_outputs = sann.forward_pass(sample_ann, inputs)
    # Outputs should change after backpropagation.
    assert new_outputs != outputs
    # Outputs should still be valid.
    assert all(0 <= output <= 1 for output in new_outputs)


def test_train(sample_ann):
    """
    Test the training of the ANN with sample data.
    This will check if the ANN can be trained and if the weights are updated.
    """
    # Update the sample ANN to have some known weights and biases.
    sample_ann[0][0]["weights"] = [0.5, 0.2, 0.8]
    sample_ann[0][0]["bias"] = 0

    # Sample training data: list of tuples (inputs, expected_output)
    train_data = [
        ([0.5, 0.2, 0.8], [1, 0]),
        ([0.1, 0.4, 0.6], [0, 1]),
    ]

    # Mock the log function to avoid printing during tests
    log = MagicMock()

    # Train the ANN
    sann.train(sample_ann, train_data, epochs=10, learning_rate=0.1, log=log)

    # Check if weights have been updated after training.
    assert sample_ann[0][0]["weights"] != [0.5, 0.2, 0.8]
    # Check if the bias has been updated.
    assert sample_ann[0][0]["bias"] != 0
    # Check the log function was called
    log.assert_called()
