"""
Test file for the sann package, using PyTest.
"""

import sann
import pytest


@pytest.fixture
def sample_ann():
    """
    Fixture to create a sample artificial neural network (ANN) for testing.
    This ANN has 2 layers: an input layer with 3 nodes and an output layer with 2 nodes.
    """
    return sann.create_ann([3, 2])


def test_create_ann(sample_ann):
    """
    Test the creation of an ANN with 2 layers.
    The first layer should have 3 nodes and the second layer should have 2 nodes.
    """
    assert len(sample_ann) == 1  # One layer after input layer
    assert len(sample_ann[0]) == 2  # Two nodes in the output layer
    assert (
        len(sample_ann[0][0]["weights"]) == 3
    )  # A node has weights for 3 inputs
    assert "bias" in sample_ann[0][0]  # A node should have a bias


def test_forward_pass(sample_ann):
    """
    Test the forward pass through the ANN with sample inputs.
    The inputs should be a list of 3 values corresponding to the input layer.
    """
    inputs = [0.5, 0.2, 0.8]
    outputs = sann.forward_pass(sample_ann, inputs)

    # Check that the output is a list of length equal to the number of nodes in the output layer
    assert len(outputs) == 2
    assert all(
        0 <= output <= 1 for output in outputs
    )  # Outputs should be between 0 and 1
