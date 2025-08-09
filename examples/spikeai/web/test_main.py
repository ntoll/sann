#!/usr/bin/env python3
"""
Test script to verify the bot logic without the web interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bot import Bot, BotWorld
import math

class TestBot(Bot):
    def __init__(self, world, shape="🔺", leave_trail=False):
        super().__init__()
        self.world = world
        self.shape = shape
        self.leave_trail = leave_trail

    def detect_colour(self):
        # This would be implemented by the world
        self.colour_reading = self.world.get_colour_ahead(self)

    def detect_distance(self):
        # This would be implemented by the world
        self.distance_reading = self.world.get_distance_ahead(self)

class TestWorld(BotWorld):
    def __init__(self, width=20, height=20):
        super().__init__(width, height)
        self.bots = []
        self.obstacles = {}

    def add_bot(self, bot, x, y, angle=0.0):
        self.bots.append({'bot': bot, 'x': x, 'y': y, 'angle': angle % 360})

    def add_obstacle(self, x, y, kind):
        if kind in [self.WALL, self.RED, self.GREEN, self.BLUE, self.YELLOW]:
            self.obstacles[(x, y)] = kind

    def get_colour_ahead(self, bot):
        for b in self.bots:
            if b['bot'] is bot:
                angle = math.radians(b['angle'])
                dx, dy = round(math.sin(angle)), -round(math.cos(angle))
                nx, ny = b['x'] + dx, b['y'] + dy
                colour_map = {
                    self.RED: 1,
                    self.GREEN: 2,
                    self.BLUE: 3,
                    self.YELLOW: 4,
                }
                cell = self.obstacles.get((nx, ny), self.EMPTY)
                return colour_map.get(cell, 0)
        return 0

    def get_distance_ahead(self, bot):
        for b in self.bots:
            if b['bot'] is bot:
                angle = math.radians(b['angle'])
                dx, dy = math.sin(angle), -math.cos(angle)
                for dist in range(1, 6):
                    nx = int(round(b['x'] + dx * dist))
                    ny = int(round(b['y'] + dy * dist))
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        return dist
                    if (nx, ny) in self.obstacles:
                        return dist
                    if any(other['x'] == nx and other['y'] == ny and other['bot'] != bot for other in self.bots):
                        return dist
                return 0
        return 0

    def tick(self):
        print(f"\n--- Tick ---")
        for b in self.bots:
            bot = b['bot']
            x, y = b['x'], b['y']
            angle = b['angle']

            bot.detect_world()
            print(f"Bot at ({x}, {y}) facing {angle:.1f}°")
            print(f"  Color reading: {bot.colour_reading}, Distance reading: {bot.distance_reading}")

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
            elif bot.distance_reading > 0 and bot.distance_reading <= 2:  # Close obstacle - turn around
                bot.set_motors(-1.0, 1.0)
            else:  # No obstacle detected or far away - move forward
                bot.set_motors(1.0, 1.0)

            print(f"  Motor change: L:{old_left:.1f}→{bot.left_motor:.1f}, R:{old_right:.1f}→{bot.right_motor:.1f}")

            # Calculate movement
            rotation_speed = (bot.right_motor - bot.left_motor) * 10.0
            new_angle = (angle + rotation_speed) % 360

            forward_speed = (bot.left_motor + bot.right_motor) / 2.0
            if forward_speed != 0:
                dx = math.sin(math.radians(new_angle)) * forward_speed * 2
                dy = -math.cos(math.radians(new_angle)) * forward_speed * 2
                nx, ny = x + int(round(dx)), y + int(round(dy))

                if (nx, ny) not in self.obstacles and 0 <= nx < self.width and 0 <= ny < self.height:
                    if not any(other['x'] == nx and other['y'] == ny and other['bot'] != bot for other in self.bots):
                        b['x'] = nx
                        b['y'] = ny
                        b['angle'] = new_angle
                        print(f"  Moved to ({nx}, {ny}) facing {new_angle:.1f}°")
                    else:
                        print(f"  Collision with another bot at ({nx}, {ny})")
                else:
                    b['angle'] = (angle + 180) % 360
                    print(f"  Hit obstacle/boundary, turned to {b['angle']:.1f}°")
            else:
                b['angle'] = new_angle
                print(f"  Rotated to {new_angle:.1f}°")

if __name__ == "__main__":
    # Test the bot logic
    world = TestWorld(20, 20)
    
    # Add obstacles
    world.add_obstacle(5, 5, world.RED)
    world.add_obstacle(15, 5, world.GREEN)
    world.add_obstacle(5, 15, world.BLUE)
    world.add_obstacle(15, 15, world.YELLOW)
    world.add_obstacle(10, 2, world.WALL)
    
    # Create and add bot
    bot = TestBot(world, leave_trail=True)
    world.add_bot(bot, 10, 10)
    
    print("Starting simulation...")
    print(f"World size: {world.width}x{world.height}")
    print(f"Obstacles: {world.obstacles}")
    
    # Run several ticks
    for i in range(10):
        world.tick()
        
    print("\nSimulation complete!")
