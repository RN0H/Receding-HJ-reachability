import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

# Define the environment size
ENV_SIZE = 10
DYNAMIC_OBSTACLES_FLAG = True

class Obstacle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.mass = np.pi * r**2  # assuming uniform density
        
        self.vx = np.random.uniform(-3, 3) if DYNAMIC_OBSTACLES_FLAG else 0.
        self.vy = np.random.uniform(-3, 3) if DYNAMIC_OBSTACLES_FLAG else 0.

    def move(self, dt=0.1):
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Reflect off walls
        if self.x - self.r < 0 or self.x + self.r > ENV_SIZE:
            self.vx *= -1
        if self.y - self.r < 0 or self.y + self.r > ENV_SIZE:
            self.vy *= -1

    def draw(self, ax):
        self.circle = plt.Circle((self.x, self.y), self.r, color='red', alpha=0.5)
        ax.add_patch(self.circle)
        return self.circle

    def update_patch(self):
        self.circle.center = (self.x, self.y)
        return [self.circle]
    
    def resolve_collision(self, obstacle, restitution=0.8):
        dx, dy = obstacle.x - self.x, obstacle.y - self.y
        dist = np.hypot(dx, dy)
        min_dist = self.r + obstacle.r

        if dist == 0 or dist >= min_dist:
            return  # no collision
        
        if dist < 1e-6: # Add a small epsilon to avoid near-zero division
             nx, ny = 0, 0 # Effectively no normal if too close to avoid instability
        else:
            nx, ny = dx / dist, dy / dist
        # Normalize vector

        # Resolve penetration: push obstacles apart
        overlap = min_dist - dist
        correction = 0.5 * overlap  # push each by half the overlap
        self.x -= correction * nx
        self.y -= correction * ny
        obstacle.x += correction * nx
        obstacle.y += correction * ny

        # Relative velocity
        dvx = self.vx - obstacle.vx
        dvy = self.vy - obstacle.vy
        rel_vel = dvx * nx + dvy * ny

        if rel_vel > 0:
            return  # already moving apart after position correction

        # Elastic collision with restitution (1 = perfectly elastic)
        impulse = -(1 + restitution) * rel_vel / (self.mass + obstacle.mass)
        self.vx += impulse * nx / self.mass
        self.vy += impulse * ny / self.mass
        obstacle.vx -= impulse * nx * obstacle.mass
        obstacle.vy -= impulse * ny * obstacle.mass