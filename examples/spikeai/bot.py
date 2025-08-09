"""
A naive representation of a trundle bot and its world. Used to test the neural
network's ability to learn how to navigate the world.

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
import random


class Bot:
    """
    Represents a simple bot with two motors: left and right. Each motor can
    have a value in the range of: 1000 (max forwards), 0 (stopped) or -1000
    (max backwards). The bot can move in any direction by adjusting the speed
    and direction of its motors.

    The bot also has two sensors: one to detect colour and the other to detect
    distance. The colour sensor detects one of five possible colours (red, green,
    blue, yellow, or none), while the distance sensor measures the distance to the
    nearest obstacle with six possible ranges (0 [nothing detected], 1 [very
    close], 2 [close], 3 [medium], 4 [far], or 5 [very far]).

    To make the bot useful it should first detect the world using its sensors,
    some sort of computation should be performed to determine the state of the
    bot's motors, and then the motors should be set accordingly.

    For convenience, the bot also provides an `input_layer` method that returns
    a list of inputs to a neural network. This list contains the current colour
    sensor reading and the distance sensor reading, represented as a list of
    floating-point numbers. The first five indices represent the colour sensor
    readings (0 for no colour, 1 for red, 2 for green, 3 for blue, 4 for yellow),
    and the next six indices represent the six possible states for the distance
    sensor reading.
    """

    def __init__(self):
        """
        Initialise the bot's motors and sensors.
        """
        self.left_motor = 0  # Default off.
        self.right_motor = 0  # Default off.
        self.colour_reading = 0  # No colour detected.
        self.distance_reading = 0  # Default no distance detected.

    def set_motors(self, left: float, right: float):
        """
        Set the state of the left and right motors. Valid values are between
        -1 (backward), and 0 (off), to 1 (forward).
        """
        self.left_motor = left
        self.right_motor = right
    
    def detect_world(self):
        """
        Detect readings from the colour and distance sensors.
        """
        self.detect_colour()
        self.detect_distance()
    
    def detect_colour(self):
        """
        Detect the colour in front of the bot. The colour can be one of:

        - 0 (no colour detected - move forward)
        - 1 (red - stop)
        - 2 (green - move forward)
        - 3 (blue - turn left)
        - 4 (yellow - turn right)
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def detect_distance(self):
        """
        Detect the distance to the nearest obstacle in front of the bot. The distance
        can be one of:

        - 0 (nothing detected)
        - 1 (very close)
        - 2 (close)
        - 3 (medium)
        - 4 (far)
        - 5 (very far)
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
    
    def input_layer(self):
        """
        Return the inputs to the neural network. This is a list of nodes representing
        the bot's current colour sensor reading, and distance sensor reading.
        """
        input_layer = [0.0 for _ in range(11)]  # 5 colours + 6 distance values.
        # Set colour sensor reading.
        input_layer[self.colour_reading] = 1.0
        # Set distance sensor reading.
        input_layer[self.distance_reading + 5] = 1.0
        return input_layer


class BotWorld:
    """
    Represents a simple world for the bot to navigate. The world is a grid of
    cells where the bot can move around. The world may contain obstacles which the
    bot can detect using its sensors. The world can also contain multiple bots
    that can interact with each other. The edge of the world does NOT wrap around
    and encountering the edge is considered the same as encountering an obstacle.
    """

    WALL = "🧱"
    RED = "🟥"
    GREEN = "🟩"
    BLUE = "🟦"
    YELLOW = "🟨"
    EMPTY = "⬜"
    BOT = "🔺"

    def __init__(self, width: int = 200, height: int = 200):
        """
        Initialise the world with a given width and height.
        """
        self.width = width
        self.height = height
        self.obstacles = {}  # Dictionary of obstacle positions.
        self.bots = {}  # Dictionary of bots keyed by their x and y positions.

    def add_bot(self, bot: Bot):
        """
        Add a bot to the world. The bot's position (x, y) and heading (degrees)
        is tracked by the world.

        The bot should be placed at a valid position within the world with a
        random heading.

        If the position is already occupied by another bot or an obstacle, it
        will not be added.
        """
        position_x = random.randint(1, self.width - 1)
        position_y = random.randint(1, self.height - 1)
        heading = random.randint(0, 360)
        if (position_x, position_y) not in self.obstacles and (position_x, position_y) not in self.bots:
            self.bots[(position_x, position_y)] = {
                "bot": bot,
                "heading": heading
            }
        else:
            raise ValueError("Random position already occupied by another bot or obstacle.")

    def add_obstacle(self, position_x: int, position_y: int, obstacle_type: str = "wall"):
        """
        Add an obstacle to the world at the given coordinates. The obstacle
        type can be specified (default is "wall"). The position must be within
        the bounds of the world and not already occupied by another obstacle.
        """
        if (position_x, position_y) not in self.obstacles:
            self.obstacles[(position_x, position_y)] = obstacle_type
    
    def tick(self):
        """
        Update the world by one tick. This will move all bots and check for
        collisions with obstacles or other bots.
        """
        for position, bot_info in list(self.bots.items()):
            bot = bot_info["bot"]
            # Move the bot based on its motor settings.
            bot.detect_world()
            # Get the input layer for the bot.
            inputs = bot.input_layer()
            # Here you would typically pass the inputs to a neural network to get outputs.
            # For now, we will just simulate some random movement.
            left_motor = random.choice([-1, 0, 1])
            right_motor = random.choice([-1, 0, 1])
            bot.set_motors(left_motor, right_motor)
            # Update the bot's position based on its motors (this is a simplification).
            new_x = (position[0] + left_motor) % self.width
            new_y = (position[1] + right_motor) % self.height
            new_position = (new_x, new_y)
            if new_position not in self.obstacles and new_position not in self.bots:
                del self.bots[position]
                self.bots[new_position] = {"bot": bot, "heading": bot_info["heading"]}