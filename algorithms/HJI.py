"""
Cost

"""
import jax
import jax.numpy as jnp
import numpy as np
from env.config import EnvConfig

######## Generating Value functions ###########

class HJI:

    def __init__(self, grid):
        self.grid = grid

    # --- Reintegrate Function ---

    @staticmethod
    def compute_values(local_grid, obstacles, goals, obstacle_strength, goal_strength):
        """
        Returns V(x, t=0)
        """
        initial_values = jnp.zeros(local_grid.shape)
        for goal in goals:
            goal_field = goal_strength * HJI.circle_sdf(local_grid, goal["center"], gamma=goal["radius"])
            initial_values += goal_field

        for obs in obstacles:
            obs_field = obstacle_strength * HJI.circle_sdf(local_grid, obs["center"], gamma=obs["radius"])
            initial_values += obs_field
        
        return initial_values
    
    ##### Signed Distance Functions ########
    @staticmethod
    def circle_sdf(grid, center, gamma=0.5):
        coords = grid.states[..., :2]
        dist = jnp.linalg.norm(coords - jnp.array(center), axis=-1)
        return dist - gamma  # < 0 inside, > 0 outside
    
    @staticmethod
    def gaussian_sdf(grid, center, gamma=0.7):
        coords = grid.states[..., :2]
        dist = jnp.linalg.norm(coords - jnp.array(center), axis=-1)
        return jnp.exp(- (dist**2)) 
    
    @staticmethod
    def smooth_sdf(field, scale=1.0):
        return jnp.tanh(scale * field)
    
if __name__ == "__pass__":
    ##### Test #####
    pass



