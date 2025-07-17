"""
Training script for playing the game of snake using a feedforward neural network
as defined in the `sann` module and otherwise standard Python. This script trains the
model to play the game by simulating the game environment and using the ANN to make
decisions based on the game state. Newer and better versions of the ANN are evolved
using a genetic algorithm approach.

Since this should work with MicroPython the script does not use any external libraries
other than those in the (MicroPython) Python standard library.
"""

import sys

sys.path.append("../../")  # Adjust path to import sann module
import json
import sann
from snake import SnakeWorld
from rich.progress import Progress
