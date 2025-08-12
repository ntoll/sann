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

import math


class Bot:
    """
    Represents a simple bot with two motors: left and right. Each motor can
    have a value in the range of: 1 (max forwards), 0 (stopped) or -1
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
        self.collided = False  # Default not collided.

    def set_motors(self, left: float, right: float):
        """
        Set the state of the left and right motors. Valid values are between
        -1 (backward), and 0 (off), to 1 (forward).
        """
        if not self.collided:  # Don't change motors if collided
            self.left_motor = left
            self.right_motor = right

    def detect_world(self):
        """
        Detect readings from the colour and distance sensors.
        """
        self.detect_color()
        self.detect_distance()
        self.drive()

    def detect_color(self):
        """
        Detect the colour in front of the bot. The colour can be one of:

        - 0 (no colour detected - move forward)
        - 1 (red - stop)
        - 2 (green - move forward)
        - 3 (blue - turn left)
        - 4 (yellow - turn right)
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def detect_distance(self):
        """
        Detect the distance to the nearest obstacle in front of the bot. The
        distance can be one of:

        - 0 (nothing detected)
        - 1 (very close)
        - 2 (close)
        - 3 (medium)
        - 4 (far)
        - 5 (very far)
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def drive(self):
        """
        Called immediately after the world has been detected. Update the bot's
        driving.
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def input_layer(self):
        """
        Return the inputs to the neural network. This is a list of nodes
        representingthe bot's current colour sensor reading, and distance
        sensor reading.
        """
        input_layer = [
            0.0 for _ in range(11)
        ]  # 5 colours + 6 distance values.
        # Set colour sensor reading.
        input_layer[self.colour_reading] = 1.0
        # Set distance sensor reading.
        input_layer[self.distance_reading + 5] = 1.0
        return input_layer


class SANNBot(Bot):
    """
    A bot that uses a neural network to navigate the world.
    """

    def __init__(self, world, brain):
        super().__init__()
        self.world = world
        # The ANN associated with this bot.
        self.brain = brain
        # The bot's lifespan measured in ticks.
        self.lifespan = 0
    
    def detect_color(self):
        self.colour_reading = self.world.get_color_ahead(
            self.x, self.y, self.angle
        )

    def detect_distance(self):
        self.distance_reading = self.world.get_distance_ahead(
            self.x, self.y, self.angle
        )

    def drive(self):
        """
        Update the bot's motors based on its brain's output given the current
        state of its sensors.

        There are two outputs for each wheel, one for travelling forwards, the
        other for moving backwards. The highest output for each wheel 
        determines its direction and speed.
        """
        # Run the sensors through the neural network to get an output 
        # decision.
        outputs = sann.run_network(self.brain, self.input_layer())
        # Outputs 0/1 control forwards/backwards movement of left wheel.
        if outputs[0] >= outputs[1]:
            left_wheel_output = outputs[0]
        else:
            left_wheel_output = -outputs[1]
        # Outputs 2/3 control forwards/backwards movement of right wheel.
        if outputs[2] >= outputs[3]:
            right_wheel_output = outputs[2]
        else:
            right_wheel_output = -outputs[3]
        self.set_motors(left_wheel_output, right_wheel_output)


class BotWorld:
    """
    Represents a simple virtual world for the bot to navigate. The world is a
    grid of cells where the bot can move around. The world may contain
    obstacles which the bot can detect using its sensors. The world can also
    contain multiple bots that can interact with each other. The edge of the
    world does NOT wrap around and encountering the edge is considered the
    same as encountering an obstacle.
    """

    # Obstacle types, along with how they might be represented visually.
    WALL_OBSTACLE = "🧱"
    RED_OBSTACLE = "🟥"
    GREEN_OBSTACLE = "🟩"
    BLUE_OBSTACLE = "🟦"
    YELLOW_OBSTACLE = "🟨"

    # Map of obstacle colours to their sensor readings.
    COLOR_MAP = {
        "RED": 1,
        "GREEN": 2,
        "BLUE": 3,
        "YELLOW": 4,
    }

    # Colours to be aware of.
    COLOR_OBSTACLES = {
        RED_OBSTACLE: COLOR_MAP["RED"],
        GREEN_OBSTACLE: COLOR_MAP["GREEN"],
        BLUE_OBSTACLE: COLOR_MAP["BLUE"],
        YELLOW_OBSTACLE: COLOR_MAP["YELLOW"],
    }

    def __init__(self, width: int = 200, height: int = 200):
        """
        Initialise the world with a given width and height.
        """
        self.width = width
        self.height = height
        self.obstacles = {}  # Dictionary of obstacle positions.
        self.bots = []  # a list of bots found in the world.

    async def update_world(self):
        """
        Draw the world, including all bots and obstacles.
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def add_bot(self, bot: Bot):
        """
        Add a bot to the world.
        """
        raise NotImplementedError(
            "This method should be implemented in a subclass."
        )

    def add_obstacle(self, x: int, y: int, obstacle_type: str = WALL_OBSTACLE):
        """
        Add an obstacle to the world at the given coordinates. The obstacle
        type can be specified (default is a wall).
        """
        if obstacle_type in [
            self.WALL_OBSTACLE,
            self.RED_OBSTACLE,
            self.GREEN_OBSTACLE,
            self.BLUE_OBSTACLE,
            self.YELLOW_OBSTACLE,
        ]:
            self.obstacles[(x, y)] = obstacle_type

    def get_direction_from_angle(self, angle):
        """
        Calculate the direction the bot is facing (dx, dy) from its angle.
        """
        angle_rad = math.radians(angle)
        return math.sin(angle_rad), -math.cos(angle_rad)

    def get_color_ahead(self, x, y, angle):
        """
        Get the colour of any obstacles in front of the bot.
        """
        dx, dy = self.get_direction_from_angle(angle)
        # Scan the range of cells ahead, starting from teh closest.
        for dist in range(1, 6):
            # Get the target cell coordinates.
            nx = int(round(x + dx * dist))
            ny = int(round(y + dy * dist))
            # Check it's within bounds.
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                break
            # Get the content of the cell at the target coordinates.
            cell = self.obstacles.get((nx, ny), None)
            # Check if the cell is a colored obstacle.
            if cell in self.COLOR_OBSTACLES:
                # Yes, so return this (the closest colour reading).
                return self.COLOR_OBSTACLES[cell]
            # Check if the cell is a non-colored obstacle (like a wall)
            if cell is not None:
                # Yes, so stop scanning. ;-)
                break
        # If we get here, there's no coloured obstacle in range.
        return 0

    def get_distance_ahead(self, x, y, angle):
        """
        Get the distance to any obstacles in front of the bot.
        """
        dx, dy = self.get_direction_from_angle(angle)
        # Scan the range of cells ahead, starting from the closest.
        for dist in range(1, 6):
            # Target cell coordinates.
            nx = int(round(x + dx * dist))
            ny = int(round(y + dy * dist))
            # Check if the target cell is within bounds.
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                return dist
            # Check if the target cell is occupied by an obstacle.
            if (nx, ny) in self.obstacles:
                return dist
            # Check for other bots at this position.
            if any(other.x == nx and other.y == ny for other in self.bots):
                return dist
        # If we get here, there's no obstacle in range.
        return 0

    async def tick(self):
        """
        Update the world by one tick. This will move all bots and check for
        collisions with obstacles or other bots.
        """
        for bot in self.bots:
            # Move the bot based on its motor settings.
            bot.detect_world()
        await self.update_world()
