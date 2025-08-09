from bot import Bot, BotWorld
from pyscript.web import page
import asyncio
import math

class InteractiveBot(Bot):
    def __init__(self, world, shape="🔺", leave_trail=False):
        super().__init__()
        self.world = world
        self.shape = shape
        self.leave_trail = leave_trail

    def detect_colour(self):
        self.colour_reading = self.world.get_colour_ahead(self)

    def detect_distance(self):
        self.distance_reading = self.world.get_distance_ahead(self)


class BotWorld(BotWorld):

    def __init__(self, width: int = 200, height: int = 200):
        super().__init__(width, height)
        self.canvas = page.find("#botworld-canvas")[0]
        self.ctx = self.canvas.getContext("2d")
        self.bots = []

    def add_bot(self, bot: InteractiveBot, x: int, y: int, angle: float = 0.0):
        self.bots.append({'bot': bot, 'x': x, 'y': y, 'angle': angle % 360})

    def add_obstacle(self, x: int, y: int, kind: str):
        if kind in [self.WALL, self.RED, self.GREEN, self.BLUE, self.YELLOW]:
            self.obstacles[(x, y)] = kind

    def get_colour_ahead(self, bot: InteractiveBot):
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

    def get_distance_ahead(self, bot: InteractiveBot):
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

    async def tick(self):
        old_positions = {b['bot']: (b['x'], b['y']) for b in self.bots}
        trails = {}

        for b in self.bots:
            bot = b['bot']
            x, y = b['x'], b['y']
            angle = b['angle']

            bot.detect_world()

            # Improved bot behavior based on sensor readings
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

            # Calculate new position based on motor values
            rotation_speed = (bot.right_motor - bot.left_motor) * 10.0
            new_angle = (angle + rotation_speed) % 360

            forward_speed = (bot.left_motor + bot.right_motor) / 2.0
            if forward_speed != 0:  # Only move if there's forward speed
                dx = math.sin(math.radians(new_angle)) * forward_speed * 2
                dy = -math.cos(math.radians(new_angle)) * forward_speed * 2
                nx, ny = x + int(round(dx)), y + int(round(dy))

                # Check bounds and collisions
                if (nx, ny) not in self.obstacles and 0 <= nx < self.width and 0 <= ny < self.height:
                    if not any(other['x'] == nx and other['y'] == ny and other['bot'] != bot for other in self.bots):
                        if bot.leave_trail:
                            trails[(x, y)] = "✨"
                        b['x'] = nx
                        b['y'] = ny
                        b['angle'] = new_angle
                    else:
                        self.handle_collision((nx, ny))
                else:
                    # Hit boundary or obstacle, turn around
                    b['angle'] = (angle + 180) % 360
            else:
                # Just update angle if not moving forward
                b['angle'] = new_angle

        await self.animate_movement(old_positions, self.bots, trails)
        self.trails = trails

    async def animate_movement(self, old_positions, new_bots, trails):
        if not self.canvas:
            return
        tile_size = 20
        base_steps = 10

        for step in range(1, base_steps + 1):
            self.ctx.clearRect(0, 0, self.canvas.width, self.canvas.height)
            self.draw_static(self.ctx, tile_size, trails)

            for info in new_bots:
                bot = info['bot']
                new_x, new_y = info['x'], info['y']
                old_x, old_y = old_positions.get(bot, (new_x, new_y))
                speed = abs((bot.left_motor + bot.right_motor) / 2.0)
                angle_deg = info['angle']

                # Interpolate position for smooth animation
                draw_x = old_x + (new_x - old_x) * (step / base_steps)
                draw_y = old_y + (new_y - old_y) * (step / base_steps)

                # Convert grid coordinates to canvas coordinates
                canvas_x = draw_x * tile_size + tile_size // 2
                canvas_y = draw_y * tile_size + tile_size // 2

                # Save context for rotation
                self.ctx.save()
                
                # Move to bot position and rotate
                self.ctx.translate(canvas_x, canvas_y)
                self.ctx.rotate(math.radians(angle_deg))
                
                # Set font properties
                self.ctx.font = f"{tile_size-4}px serif"
                self.ctx.textAlign = "center"
                self.ctx.textBaseline = "middle"
                
                # Draw the bot shape (centered due to textAlign and textBaseline)
                self.ctx.fillText(bot.shape, 0, 0)
                
                # Restore context
                self.ctx.restore()

            await asyncio.sleep(0.02 * (1.5 - min(speed, 1.5)))

    def draw_static(self, ctx, tile_size, trails={}):
        # Set canvas dimensions
        self.canvas.width = self.width * tile_size
        self.canvas.height = self.height * tile_size
        
        # Set font properties for static elements
        ctx.font = f"{tile_size-4}px serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"

        # Draw obstacles
        for (x, y), kind in self.obstacles.items():
            canvas_x = x * tile_size + tile_size // 2
            canvas_y = y * tile_size + tile_size // 2
            ctx.fillText(kind, canvas_x, canvas_y)

        # Draw trail symbols
        for (x, y), symbol in trails.items():
            canvas_x = x * tile_size + tile_size // 2
            canvas_y = y * tile_size + tile_size // 2
            ctx.fillText(symbol, canvas_x, canvas_y)

    def handle_collision(self, pos):
        if not self.canvas:
            return
        x, y = pos
        tile_size = 20
        canvas_x = x * tile_size + tile_size // 2
        canvas_y = y * tile_size + tile_size // 2
        
        self.ctx.font = f"{tile_size-4}px serif"
        self.ctx.textAlign = "center"
        self.ctx.textBaseline = "middle"
        self.ctx.fillText("💥", canvas_x, canvas_y)
        #audio = document.createElement("audio")
        #audio.src = "/static/collision.mp3"
        #audio.play()




bw = BotWorld(20, 20)

# Add some obstacles for the bot to interact with
bw.add_obstacle(5, 5, bw.RED)    # Red obstacle - bot should stop
bw.add_obstacle(15, 5, bw.GREEN)  # Green obstacle - bot should continue
bw.add_obstacle(5, 15, bw.BLUE)   # Blue obstacle - bot should turn left
bw.add_obstacle(15, 15, bw.YELLOW) # Yellow obstacle - bot should turn right
bw.add_obstacle(10, 2, bw.WALL)   # Wall obstacle

bot = InteractiveBot(bw, leave_trail=True)

# Place bot within the world bounds (20x20)
bw.add_bot(bot, 10, 10)

async def main():
    while True:
        await bw.tick()

async def run_simulation():
    """Run the bot simulation"""
    print("Bot world initialized. Starting main loop...")
    await main()

# Start the simulation
import asyncio
asyncio.create_task(run_simulation())