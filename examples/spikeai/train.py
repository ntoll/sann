"""
This script uses SANN to train a bot to navigate a simple virtual world. The
resulting model can be used to control a SPIKE Prime based bot with similar
capabilities.

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
import sys

sys.path.append("../../")  # Adjust path to import sann module

import sann
import json
import math
import rich
import random
from bot import SANNBot, BotWorld
from rich.progress import Progress


# The number of ANNs in each generation.
population_size = 500
# The maximum number of generations to train for.
max_generations = 1000
# The current highest fitness score.
current_max_fitness = 0
# The number of generations since the last fitness improvement.
fitness_last_updated = 0
# Fitness plateau duration (in generations). If the fitness has not improved
# for this many generations, training will be halted.
fitness_plateau_duration = 100
# The maximum number of ticks allowed in a single game.
max_game_ticks = 10000
# The name of the file to save the fittest ANN.
fittest_ann_file = "fittest_ann.json"


class TrainingWorld(BotWorld):
    """
    A virtual world for training bots.
    """

    def add_bot(self, bot: SANNBot, x: int, y: int, angle: float = 0.0):
        """
        Annotate the bot with a bunch of implementation details for the sake
        of convenience in the web world.
        """
        bot.x = x
        bot.y = y
        bot.angle = angle % 360
        self.bots.append(bot)

    def update_world(self):
        """
        Update the state of the world by moving all bots and checking for
        collisions.
        """
        # Remove dead bots
        self.bots = [bot for bot in self.bots if not bot.collided]
        for bot in self.bots:
            # Update lifespan
            bot.lifespan += 1
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
                    # Set new position.
                    bot.x = nx
                    bot.y = ny
                else:
                    # Bang!
                    bot.collided = True
            # Update bot's angle based on rotation speed.
            bot.angle = (bot.angle + rotation_speed) % 360


def fitness_function(ann, current_population):
    """
    Calculate the fitness of a bot's ann based on its performance in the
    training world. The top 4 fittest bots from the current_population
    are also added to the world, to give the current ann based bot others
    to avoid.
    """
    # Create the training world and populate it with static obstacles.
    tw = TrainingWorld(40, 40)
    # Walls around the edges.
    for x in range(40):
        tw.add_obstacle(x, 0)
        tw.add_obstacle(x, 39)
    for y in range(40):
        tw.add_obstacle(0, y)
        tw.add_obstacle(39, y)
    # Add a random amount of randomly placed walls into the world while
    # keeping track of the positions of the walls so bots cannot be added
    # to the same position.
    wall_positions = set()
    for _ in range(random.randint(10, 20)):
        x = random.randint(1, 38)
        y = random.randint(1, 38)
        wall_positions.add((x, y))
    for pos in wall_positions:
        tw.add_obstacle(*pos)
    # Add the bot, whose fitness we're checking, to the world, whilst
    # avoiding the walls.
    while True:
        x = random.randint(1, 38)
        y = random.randint(1, 38)
        if (x, y) not in wall_positions:
            break
    bot = SANNBot(tw, ann)
    tw.add_bot(bot, x, y)
    # Now add the top 4 fittest bots from the current population to the world.
    for ann in current_population[:4]:
        while True:
            x = random.randint(1, 38)
            y = random.randint(1, 38)
            if (x, y) not in wall_positions:
                break
        tw.add_bot(SANNBot(tw, ann), x, y)
    # Now run the world for the maximum number of ticks
    for _ in range(max_game_ticks):
        tw.update_world()
        if bot.collided:
            # No need to continue if the bot has collided with something.
            break
    # The fittest bots will survive the longest in the world by avoiding all
    # the obstacles, so the bot's lifespan is a measure of its fitness.
    return bot.lifespan


def halt_function(current_population, generation_count):
    """
    If the current population has not improved for fitness_plateau_duration
    generations, halt the training process. Or, if the generation_count
    exceeds 1000, also halt the training process.
    """
    if generation_count > max_generations:
        return True

    global current_max_fitness
    global fitness_last_updated

    # Check if the current population has improved.
    if current_population[0]["fitness"] > current_max_fitness:
        current_max_fitness = current_population[0]["fitness"]
        # Reset the fitness last updated counter.
        fitness_last_updated = 0
        if current_max_fitness == max_game_ticks:
            # Reached the max possible fitness, so stop training.
            return True
        return False  # Continue training
    else:
        # Increment the fitness last updated counter.
        fitness_last_updated += 1
        # If the fitness has not improved for fitness_plateau_duration 
        # generations, halt training.
        if fitness_last_updated > fitness_plateau_duration:
            return True
        else:
            return False

def main():
    """
    Main function to run the training process.
    """

    # Create a progress bar for visual feedback.
    with Progress() as progress:
        evolution_task = progress.add_task(
            "Training...", total=max_generations
        )

        def handle_log(data):
            progress.update(
                evolution_task,
                advance=1,
                description=f"Max fitness: {current_max_fitness}",
            )

        population = sann.evolve(
            layers=[11, 8, 4],
            population_size=population_size,
            fitness_function=fitness_function,
            halt_function=halt_function,
            log=handle_log,
        )

    # Save the fittest ANN to a file.
    with open(fittest_ann_file, "w") as f:
        ann = sann.clean_network(population[0])
        json.dump(ann, f, indent=2)
        rich.print(
            f"[green]Fittest ANN ({current_max_fitness}) saved to: [bold]{fittest_ann_file}[/bold][/green]"
        )


if __name__ == "__main__":
    main()