from bot import Bot, BotWorld
from pyscript.web import page
import asyncio
import math
import random


class WebBot(Bot):
    """
    A bot that works within the virtual world defined by a WebBotWorld instance.
    Bots based on this class are intended to run in a web browser environment.
    """

    def __init__(self, world):
        super().__init__()
        self.world = world
        self.rotate_direction = None

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
        Very stupid hard coded rules for driving the bot.
        """
        if self.colour_reading == self.world.COLOR_MAP["RED"]:  # Red - stop
            self.set_motors(0.0, 0.0)
        elif (
            self.colour_reading == self.world.COLOR_MAP["GREEN"]
        ):  # Green - move forward
            self.set_motors(1.0, 1.0)
        elif (
            self.colour_reading == self.world.COLOR_MAP["BLUE"]
        ):  # Blue - turn left
            self.set_motors(-0.5, 1.0)
        elif (
            self.colour_reading == self.world.COLOR_MAP["YELLOW"]
        ):  # Yellow - turn right
            self.set_motors(1.0, -0.5)
        elif (
            self.distance_reading > 0 and self.distance_reading <= 2
        ):  # Close obstacle - turn around
            if self.rotate_direction is None:
                self.rotate_direction = random.choice(
                    [(-1.0, 1.0), (1.0, -1.0)]
                )
            self.set_motors(*self.rotate_direction)
        else:  # No obstacle detected or far away - move forward
            self.rotate_direction = None  # Reset rotation direction
            self.set_motors(1.0, 1.0)


class WebBotWorld(BotWorld):
    """
    A web-based implementation of the bot world that includes a canvas for
    rendering the bots and their environment.
    """

    def __init__(self, width: int = 200, height: int = 200):
        """
        Add a bunch of web-specific initialization code here.
        """
        super().__init__(width, height)
        self.canvas = page.find("#botworld-canvas")[0]
        self.ctx = self.canvas.getContext("2d")
        self.trails = {}
        self.trail_max_length = 12

    def add_bot(
        self,
        bot: WebBot,
        x: int,
        y: int,
        angle: float = 0.0,
        color: list = (0, 100, 255),
    ):
        """
        Annotate the bot with a bunch of implementation details for the sake
        of convenience in the web world.
        """
        bot.x = x
        bot.y = y
        bot.angle = angle % 360
        bot.color = color
        self.bots.append(bot)
        # List of (x, y, age) tuples for each circle on the trail.
        self.trails[bot] = []

    async def update_world(self):
        """
        Update the state of the world by moving all bots and checking for
        collisions.
        """
        # Required for animating to the new state.
        old_positions = {}
        # Update the bots...
        for bot in self.bots:
            # Add bot's current position to old_positions from where it will
            # be animated.
            old_positions[bot] = (bot.x, bot.y)
            # Calculate new position/state based on motor values.
            rotation_speed = (bot.right_motor - bot.left_motor) * 10.0
            forward_speed = (bot.left_motor + bot.right_motor) / 2.0
            # Use the forward speed to update the bot's position.
            if forward_speed != 0:
                dx = math.sin(math.radians(bot.angle)) * forward_speed * 2
                dy = -math.cos(math.radians(bot.angle)) * forward_speed * 2
                nx, ny = bot.x + int(round(dx)), bot.y + int(round(dy))
                # Check bounds and collisions
                if (
                    (nx, ny) not in self.obstacles
                    and 0 <= nx < self.width
                    and 0 <= ny < self.height
                ) or not any(
                    other.x == nx and other.y == ny and other != bot
                    for other in self.bots
                ):
                    # Handle bot trails and update with new x, y
                    self.trails[bot].append((bot.x, bot.y, 0))
                    while len(self.trails[bot]) > self.trail_max_length:
                        self.trails[bot].pop(0)
                    bot.x = nx
                    bot.y = ny
                else:
                    bot.collided = True  # BANG!
            else:
                # Still drop breadcrumb for rotation within a position.
                self.trails[bot].append((bot.x, bot.y, 0))
                while len(self.trails[bot]) > self.trail_max_length:
                    self.trails[bot].pop(0)
            # Update bot's angle based on rotation speed.
            bot.angle = (bot.angle + rotation_speed) % 360

        # Always age all trail entries regardless of movement
        for bot in self.trails:
            # Age all trail entries
            for i in range(len(self.trails[bot])):
                trail_x, trail_y, age = self.trails[bot][i]
                self.trails[bot][i] = (trail_x, trail_y, age + 1)
            # Remove fully faded trail entries (age > trail_max_length means
            # opacity <= 0)
            self.trails[bot] = [
                (x, y, age)
                for x, y, age in self.trails[bot]
                if age <= self.trail_max_length
            ]
        # Now take the old positions and animate to the new positions found in
        # self.bots.
        await self.animate_movement(old_positions, self.bots)

    async def animate_movement(self, old_positions, new_bots):
        """
        Animate the movement of bots from their old positions to their new
        positions.

        Mostly written by an LLM with prompting from a human (canvas animation
        is not something I know about).

        But it seems to work!
        """
        tile_size = 20
        base_steps = 10
        # Move the bot by a small amount each step, with base_steps being the
        # number of steps to move to the new position.
        for step in range(1, base_steps + 1):
            # Clear the canvas.
            self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
            # And draw the static elements.
            self.draw_static(self.ctx, tile_size)
            # Re-draw each bot at its new "step" position.
            for bot in new_bots:
                new_x, new_y = bot.x, bot.y
                old_x, old_y = old_positions.get(bot, (new_x, new_y))
                angle_deg = bot.angle
                # Interpolate position for smooth animation.
                draw_x = old_x + (new_x - old_x) * (step / base_steps)
                draw_y = old_y + (new_y - old_y) * (step / base_steps)
                # Convert grid coordinates to canvas coordinates.
                canvas_x = draw_x * tile_size + tile_size // 2
                canvas_y = draw_y * tile_size + tile_size // 2
                # Draw sensor line first (so it appears underneath the bot)
                self.draw_sensor_line(
                    canvas_x, canvas_y, angle_deg, bot, tile_size
                )
                # The following code is the LLM's main contribution. No idea
                # what it's doing, but it seems to work. ;-)
                # Save context for rotation.
                self.ctx.save()
                # Move to bot position and rotate.
                self.ctx.translate(canvas_x, canvas_y)
                self.ctx.rotate(math.radians(angle_deg))
                # Draw custom bot shape that looks like it has two motors.
                self.draw_bot_shape(tile_size, bot)
                # Restore context.
                self.ctx.restore()

            # Sleep for a consistent amount so the animation appears smooth,
            # regardless of bot speed.
            await asyncio.sleep(0.01)

    def draw_bot_shape(self, tile_size, bot):
        """
        Draw a bot shape that looks like it has two motors seen from above.

        Mostly created by an LLM with colour features added by a human.
        """
        ctx = self.ctx
        # Size relative to tile, increased for better visibility but may
        # look like there are overlapping elements.
        size = tile_size * 1.4
        # Main body (rectangle).
        ctx.fillStyle = "#333333"  # Dark gray body
        ctx.fillRect(-size / 4, -size / 3, size / 2, size * 0.6)
        # Left motor (wheel).
        ctx.fillStyle = "#666666"  # Lighter gray for motors
        ctx.fillRect(-size / 3, -size / 4, size / 8, size / 2)
        # Right motor (wheel).
        ctx.fillRect(size / 4, -size / 4, size / 8, size / 2)
        # Front direction indicator (small rectangle at front),
        # indicating the bot's colour.
        r, g, b = bot.color
        ctx.fillStyle = f"rgb({r}, {g}, {b})"
        ctx.fillRect(-size / 8, -size / 3, size / 4, size / 10)
        # Center dot to show rotation point
        ctx.fillStyle = "#fff"
        ctx.beginPath()
        ctx.arc(0, 0, size / 12, 0, 2 * math.pi)
        ctx.fill()

    def draw_sensor_line(self, bot_x, bot_y, angle_deg, bot, tile_size):
        """
        Draw a line showing the bot's sensor readings.

        When nothing is detected, the sensor should be a light grey line. As
        objects are detected, the closer they become, the darker the colour.

        If a colour is detected, the line should match that colour while also
        changing intensity based on distance.
        """
        max_sensor_range = 5
        # Calculate line end point based on distance reading.
        if bot.distance_reading == 0:
            # No obstacle detected, draw full length line.
            line_length = max_sensor_range * tile_size
        else:
            # Obstacle detected, the line length reflects distance to
            # detected object.
            line_length = bot.distance_reading * tile_size
        # Calculate line end coordinates.
        angle_rad = math.radians(angle_deg)
        end_x = bot_x + math.sin(angle_rad) * line_length
        end_y = bot_y - math.cos(angle_rad) * line_length
        # Determine line color based on color sensor reading.
        # The default is very light grey (RGB).
        red, green, blue = 220, 220, 220
        if bot.colour_reading == self.COLOR_MAP["RED"]:
            red, green, blue = 255, 0, 0
        elif bot.colour_reading == self.COLOR_MAP["GREEN"]:
            red, green, blue = 0, 255, 0
        elif bot.colour_reading == self.COLOR_MAP["BLUE"]:
            red, green, blue = 0, 0, 255
        elif bot.colour_reading == self.COLOR_MAP["YELLOW"]:
            red, green, blue = 255, 255, 0
        # Apply distance intensity.
        if bot.distance_reading == 0:
            effective_intensity = 1.0
        else:
            # Obstacle detected, so change the intensity depending if a color
            # is detected.
            if bot.colour_reading == 0:
                # No colour, but something else (wall, another bot etc...).
                min_intensity = 0.1  # Nearly black for close objects.
                max_intensity = 0.8  # Light gray for distant objects.
                # Closer = darker (lower intensity)
                intensity_range = max_intensity - min_intensity
                distance_factor = (bot.distance_reading - 1) / (
                    max_sensor_range - 1
                )
                effective_intensity = (
                    min_intensity + intensity_range * distance_factor
                )
            else:
                # Colour detected.
                min_intensity = (
                    0.3  # Minimum intensity for distant colored objects
                )
                max_intensity = (
                    1.0  # Maximum intensity for close colored objects
                )
                # Closer = brighter (higher intensity)
                intensity_range = max_intensity - min_intensity
                distance_factor = (max_sensor_range - bot.distance_reading) / (
                    max_sensor_range - 1
                )
                effective_intensity = (
                    min_intensity + intensity_range * distance_factor
                )
        # Apply the effective intensity to the RGB values.
        red = int(red * effective_intensity)
        green = int(green * effective_intensity)
        blue = int(blue * effective_intensity)
        # Draw the actual sensor line.
        self.ctx.save()
        self.ctx.strokeStyle = f"rgb({red}, {green}, {blue})"
        self.ctx.lineWidth = 1
        self.ctx.beginPath()
        self.ctx.moveTo(bot_x, bot_y)
        self.ctx.lineTo(end_x, end_y)
        self.ctx.stroke()
        self.ctx.restore()

    def draw_static(self, ctx, tile_size):
        """
        Draw all the static / unmoving objects in the world.
        """
        # Set canvas dimensions
        self.canvas.width = self.width * tile_size
        self.canvas.height = self.height * tile_size
        # Set font properties for static elements emoji.
        ctx.font = f"{tile_size-4}px serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        # Draw obstacles
        for (x, y), kind in self.obstacles.items():
            canvas_x = x * tile_size + tile_size // 2
            canvas_y = y * tile_size + tile_size // 2
            ctx.fillText(kind, canvas_x, canvas_y)
        # Draw continuous trail lines with fading effect at the end.
        self.draw_trail_breadcrumbs(ctx, tile_size)

    def draw_trail_breadcrumbs(self, ctx, tile_size):
        """
        Draw breadcrumb trail points for each bot with fading effect.
        """
        for bot in self.bots:
            # Get the historical trail points for the bot.
            trail_points = self.trails[bot]
            if len(trail_points) < 1:
                continue
            # Get bot-specific color
            r, g, b = bot.color
            # Draw each breadcrumb point
            for x, y, age in trail_points:
                # Calculate opacity based on age (adjusted for longer trail)
                opacity = max(0.0, 1.0 - (age * 2.0 / self.trail_max_length))
                # Skip drawing if fully transparent
                if opacity <= 0.0:
                    continue
                # Convert grid coordinates to canvas coordinates
                canvas_x = x * tile_size + tile_size // 2
                canvas_y = y * tile_size + tile_size // 2
                # Draw breadcrumbs
                ctx.save()
                ctx.fillStyle = f"rgba({r}, {g}, {b}, {opacity})"
                ctx.beginPath()
                ctx.arc(canvas_x, canvas_y, 1.5, 0, 2 * math.pi)
                ctx.fill()
                ctx.restore()


# Let there be light!
bw = WebBotWorld(40, 40)


# Create irregular walls around the perimeter with protrusions and concaves
# along with some interior walls for added complexity.

# Top edge - with irregular pattern
for x in range(40):
    bw.add_obstacle(x, 0, bw.WALL_OBSTACLE)
    if x in [5, 6, 7, 15, 16, 20, 21, 22, 30, 31]:
        bw.add_obstacle(x, 1, bw.WALL_OBSTACLE)
    if x in [10, 11, 25, 26, 35]:
        bw.add_obstacle(x, 2, bw.WALL_OBSTACLE)

# Bottom edge - with different irregular pattern
for x in range(40):
    bw.add_obstacle(x, 39, bw.WALL_OBSTACLE)
    if x in [3, 4, 8, 9, 18, 19, 28, 29, 33, 34]:
        bw.add_obstacle(x, 38, bw.WALL_OBSTACLE)
    if x in [12, 13, 23, 24, 37]:
        bw.add_obstacle(x, 37, bw.WALL_OBSTACLE)

# Left edge - with irregular pattern
for y in range(40):
    bw.add_obstacle(0, y, bw.WALL_OBSTACLE)
    if y in [4, 5, 12, 13, 14, 22, 23, 32, 33]:
        bw.add_obstacle(1, y, bw.WALL_OBSTACLE)
    if y in [8, 18, 28]:
        bw.add_obstacle(2, y, bw.WALL_OBSTACLE)

# Right edge - with different irregular pattern
for y in range(40):
    bw.add_obstacle(39, y, bw.WALL_OBSTACLE)
    if y in [6, 7, 16, 17, 24, 25, 26, 34, 35]:
        bw.add_obstacle(38, y, bw.WALL_OBSTACLE)
    if y in [10, 20, 30]:
        bw.add_obstacle(37, y, bw.WALL_OBSTACLE)

# Add some corner reinforcements and interesting shapes
# Top-left corner extension
bw.add_obstacle(1, 1, bw.WALL_OBSTACLE)
bw.add_obstacle(2, 1, bw.WALL_OBSTACLE)
bw.add_obstacle(1, 2, bw.WALL_OBSTACLE)

# Top-right corner extension
bw.add_obstacle(38, 1, bw.WALL_OBSTACLE)
bw.add_obstacle(37, 1, bw.WALL_OBSTACLE)
bw.add_obstacle(38, 2, bw.WALL_OBSTACLE)

# Bottom-left corner extension
bw.add_obstacle(1, 38, bw.WALL_OBSTACLE)
bw.add_obstacle(2, 38, bw.WALL_OBSTACLE)
bw.add_obstacle(1, 37, bw.WALL_OBSTACLE)

# Bottom-right corner extension
bw.add_obstacle(38, 38, bw.WALL_OBSTACLE)
bw.add_obstacle(37, 38, bw.WALL_OBSTACLE)
bw.add_obstacle(38, 37, bw.WALL_OBSTACLE)

# Add some interior walls to create a more interesting environment
# Vertical walls
for y in range(10, 20):
    bw.add_obstacle(10, y, bw.WALL_OBSTACLE)
    bw.add_obstacle(30, y, bw.WALL_OBSTACLE)

# Horizontal walls
for x in range(15, 25):
    bw.add_obstacle(x, 15, bw.WALL_OBSTACLE)
    bw.add_obstacle(x, 25, bw.WALL_OBSTACLE)

# Create some wall corners and obstacles
bw.add_obstacle(5, 5, bw.WALL_OBSTACLE)
bw.add_obstacle(6, 5, bw.WALL_OBSTACLE)
bw.add_obstacle(5, 6, bw.WALL_OBSTACLE)

bw.add_obstacle(34, 34, bw.WALL_OBSTACLE)
bw.add_obstacle(35, 34, bw.WALL_OBSTACLE)
bw.add_obstacle(34, 35, bw.WALL_OBSTACLE)

# Add some colored obstacles for the bots to interact with
bw.add_obstacle(8, 8, bw.RED_OBSTACLE)  # bot should stop
bw.add_obstacle(32, 8, bw.GREEN_OBSTACLE)  # bot should continue
bw.add_obstacle(8, 32, bw.BLUE_OBSTACLE)  # bot should turn left
bw.add_obstacle(32, 32, bw.YELLOW_OBSTACLE)  # bot should turn right

# Add a red obstacle right in front of the center bot for testing
bw.add_obstacle(20, 19, bw.RED_OBSTACLE)


# Add some bots!
bot = WebBot(bw)

# Place bot in the center of the larger world
bw.add_bot(bot, 20, 20, 0, (0, 100, 255))  # Explicitly set angle to 0 (north)

# Add three more bots in different locations
bot2 = WebBot(bw)
bw.add_bot(bot2, 10, 10, 45, (255, 100, 0))  # Top-left area, facing northeast

bot3 = WebBot(bw)
bw.add_bot(
    bot3, 30, 10, 135, (0, 200, 100)
)  # Top-right area, facing southeast

bot4 = WebBot(bw)
bw.add_bot(bot4, 15, 30, 270, (200, 0, 200))  # Bottom area, facing west


async def main():
    while True:
        await bw.tick()


print("Bot world initialized. Starting main loop...")
await main()
