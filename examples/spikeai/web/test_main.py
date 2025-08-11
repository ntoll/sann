#!/usr/bin/env python3
"""
Test script to verify the bot logic without the web interface
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import Bot, WebBotWorld
import math


class TestBot(Bot):
    def __init__(self, world, shape="🔺", leave_trail=False):
        super().__init__()
        self.world = world
        self.shape = shape
        self.leave_trail = leave_trail

    def detect_color(self):
        # This would be implemented by the world
        self.colour_reading = self.world.get_colour_ahead(self)

    def detect_distance(self):
        # This would be implemented by the world
        self.distance_reading = self.world.get_distance_ahead(self)


class TestWorld(WebBotWorld):
    def __init__(self, width=20, height=20):
        super().__init__(width, height)
        self.bots = []
        self.obstacles = {}
        self.trails = {}  # Store trail history
        self.trail_max_length = 8  # Halved trail length for faster fading

    def add_bot(self, bot, x, y, angle=0.0):
        self.bots.append({"bot": bot, "x": x, "y": y, "angle": angle % 360})

    def add_obstacle(self, x, y, kind):
        if kind in [self.WALL, self.RED, self.GREEN, self.BLUE, self.YELLOW]:
            self.obstacles[(x, y)] = kind

    def get_colour_ahead(self, bot):
        for b in self.bots:
            if b["bot"] is bot:
                angle = math.radians(b["angle"])
                dx, dy = round(math.sin(angle)), -round(math.cos(angle))
                nx, ny = b["x"] + dx, b["y"] + dy
                colour_map = {
                    self.RED: 1,
                    self.GREEN: 2,
                    self.BLUE: 3,
                    self.YELLOW: 4,
                }
                cell = self.obstacles.get(
                    (nx, ny), None
                )  # Use None instead of EMPTY
                return colour_map.get(cell, 0)
        return 0

    def get_distance_ahead(self, bot):
        for b in self.bots:
            if b["bot"] is bot:
                angle = math.radians(b["angle"])
                dx, dy = math.sin(angle), -math.cos(angle)
                for dist in range(1, 6):
                    nx = int(round(b["x"] + dx * dist))
                    ny = int(round(b["y"] + dy * dist))
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        return dist
                    if (nx, ny) in self.obstacles:
                        return dist
                    if any(
                        other["x"] == nx
                        and other["y"] == ny
                        and other["bot"] != bot
                        for other in self.bots
                    ):
                        return dist
                return 0
        return 0

    def tick(self):
        print(f"\n--- Tick ---")

        # Initialize trail history for new bots
        for b in self.bots:
            bot = b["bot"]
            if bot not in self.trails:
                self.trails[bot] = []

        for b in self.bots:
            bot = b["bot"]
            x, y = b["x"], b["y"]
            angle = b["angle"]

            bot.detect_world()
            print(f"Bot at ({x}, {y}) facing {angle:.1f}°")
            print(
                f"  Color reading: {bot.colour_reading}, Distance reading: {bot.distance_reading}"
            )

            # Show sensor line visualization info
            if bot.distance_reading == 0:
                sensor_desc = "full range (light)"
            else:
                intensity = 1.0 - (bot.distance_reading - 1) / 4
                sensor_desc = f"{bot.distance_reading} tiles (intensity: {intensity:.2f})"

            color_names = ["none/gray", "red", "green", "blue", "yellow"]
            color_name = (
                color_names[bot.colour_reading]
                if bot.colour_reading < len(color_names)
                else "unknown"
            )
            print(f"  Sensor line: {color_name} color, {sensor_desc}")

            # Improved bot behavior
            old_left, old_right = bot.left_motor, bot.right_motor
            if bot.colour_reading == 1:  # Red - stop
                bot.set_motors(0.0, 0.0)
            elif bot.colour_reading == 2:  # Green - move forward
                bot.set_motors(1.0, 1.0)
            elif bot.colour_reading == 3:  # Blue - turn left
                bot.set_motors(-0.5, 1.0)
            elif bot.colour_reading == 4:  # Yellow - turn right
                bot.set_motors(1.0, -0.5)
            elif (
                bot.distance_reading > 0 and bot.distance_reading <= 2
            ):  # Close obstacle - turn around
                bot.set_motors(-1.0, 1.0)
            else:  # No obstacle detected or far away - move forward
                bot.set_motors(1.0, 1.0)

            print(
                f"  Motor change: L:{old_left:.1f}→{bot.left_motor:.1f}, R:{old_right:.1f}→{bot.right_motor:.1f}"
            )

            # Calculate movement
            rotation_speed = (bot.right_motor - bot.left_motor) * 10.0
            new_angle = (angle + rotation_speed) % 360

            forward_speed = (bot.left_motor + bot.right_motor) / 2.0
            if forward_speed != 0:
                dx = math.sin(math.radians(new_angle)) * forward_speed * 2
                dy = -math.cos(math.radians(new_angle)) * forward_speed * 2
                nx, ny = x + int(round(dx)), y + int(round(dy))

                if (
                    (nx, ny) not in self.obstacles
                    and 0 <= nx < self.width
                    and 0 <= ny < self.height
                ):
                    if not any(
                        other["x"] == nx
                        and other["y"] == ny
                        and other["bot"] != bot
                        for other in self.bots
                    ):
                        if bot.leave_trail:
                            trail_marker = self.get_trail_marker(angle)
                            self.trails[bot].append((x, y, angle, 0))
                            if len(self.trails[bot]) > self.trail_max_length:
                                self.trails[bot].pop(0)
                            print(
                                f"  Trail: Added position ({x}, {y}). Trail length: {len(self.trails[bot])}"
                            )
                        b["x"] = nx
                        b["y"] = ny
                        b["angle"] = new_angle
                        print(
                            f"  Moved to ({nx}, {ny}) facing {new_angle:.1f}°"
                        )
                    else:
                        print(f"  Collision with another bot at ({nx}, {ny})")
                else:
                    b["angle"] = (angle + 180) % 360
                    print(
                        f"  Hit obstacle/boundary, turned to {b['angle']:.1f}°"
                    )
            else:
                b["angle"] = new_angle
                print(f"  Rotated to {new_angle:.1f}°")

        # Always age all trail entries regardless of movement
        for bot in self.trails:
            # Age all trail entries
            for i in range(len(self.trails[bot])):
                x, y, angle, age = self.trails[bot][i]
                self.trails[bot][i] = (x, y, angle, age + 1)

            # Remove fully faded trail entries (age > trail_max_length means opacity <= 0)
            old_length = len(self.trails[bot])
            self.trails[bot] = [
                (x, y, angle, age)
                for x, y, angle, age in self.trails[bot]
                if age <= self.trail_max_length
            ]
            if len(self.trails[bot]) < old_length:
                print(
                    f"  Removed {old_length - len(self.trails[bot])} fully faded trail segments"
                )

        # Show trail aging info
        for bot in self.trails:
            if len(self.trails[bot]) > 0:
                oldest_age = self.trails[bot][-1][
                    3
                ]  # age of oldest trail segment
                newest_age = self.trails[bot][0][
                    3
                ]  # age of newest trail segment
                # Double the fade speed by multiplying age by 2
                opacity_oldest = max(
                    0.0, 1.0 - (oldest_age * 2.0 / self.trail_max_length)
                )
                opacity_newest = max(
                    0.0, 1.0 - (newest_age * 2.0 / self.trail_max_length)
                )
                print(
                    f"  Trail ages: newest={newest_age} (opacity: {opacity_newest:.2f}), oldest={oldest_age} (opacity: {opacity_oldest:.2f})"
                )
                if opacity_oldest == 0.0:
                    print(
                        f"  ↳ Oldest trail segments are now fully transparent!"
                    )

    def get_trail_marker(self, angle):
        """Return an appropriate trail marker based on the bot's direction"""
        angle = angle % 360
        if 337.5 <= angle or angle < 22.5:
            return "↑"
        elif 22.5 <= angle < 67.5:
            return "↗"
        elif 67.5 <= angle < 112.5:
            return "→"
        elif 112.5 <= angle < 157.5:
            return "↘"
        elif 157.5 <= angle < 202.5:
            return "↓"
        elif 202.5 <= angle < 247.5:
            return "↙"
        elif 247.5 <= angle < 292.5:
            return "←"
        elif 292.5 <= angle < 337.5:
            return "↖"
        else:
            return "•"


if __name__ == "__main__":
    # Test the bot logic with larger world
    world = TestWorld(40, 40)

    # Add border walls
    for x in range(40):
        world.add_obstacle(x, 0, world.WALL)
        world.add_obstacle(x, 39, world.WALL)
    for y in range(40):
        world.add_obstacle(0, y, world.WALL)
        world.add_obstacle(39, y, world.WALL)

    # Add some interior obstacles
    world.add_obstacle(8, 8, world.RED)
    world.add_obstacle(32, 8, world.GREEN)
    world.add_obstacle(8, 32, world.BLUE)
    world.add_obstacle(32, 32, world.YELLOW)

    # Create and add bot in center
    bot = TestBot(world, leave_trail=True)
    world.add_bot(bot, 20, 20, angle=0.0)

    print("Starting simulation...")
    print(f"World size: {world.width}x{world.height}")
    print(f"Bot starting position: (20, 20)")

    # Run several ticks
    for i in range(15):
        world.tick()

    print("\nSimulation complete!")
