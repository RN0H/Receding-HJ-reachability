"""
Path planner

"""

import jax.numpy as jnp
import jax
import jax.random as random

# --- MPPI Planning using Local Patch ---
def mppi_planner(dynamics, local_grid, local_values, current_state, horizon=20, num_samples=100, lambda_=1.0):
    """
    Performs MPPI planning on the current local patch.

    Args:
        dynamics: the system dynamics class (with step method)
        local_grid: hj.Grid object corresponding to local patch
        local_values: value function over the local patch
        current_state: current state (x, y, theta)
        horizon: number of timesteps to plan over
        num_samples: number of sampled trajectories
        lambda_: temperature for MPPI

    Returns:
        optimal_control: control input to apply at current timestep
    """
    
    key = random.PRNGKey(0)

    def cost_fn(state):
        
        # Find closest grid point
        indices = [jnp.abs(g - s).argmin() for g, s in zip(local_grid.coordinate_vectors, state)]
        return local_values[indices[0], indices[1], indices[2]]

    # Initialize nominal controls and trajectories
    u_nominal = jnp.zeros((horizon, dynamics.control_dim))

    def rollout_trajectory(u_traj, rng):
        states = [current_state]
        total_cost = 0.0
        for u in u_traj:
            next_state = dynamics.step(states[-1], u)
            cost = cost_fn(next_state)
            states.append(next_state)
            total_cost += cost
        return total_cost, jnp.stack(states)

    u_samples = random.normal(key, shape=(num_samples, horizon, dynamics.control_dim))
    u_samples = u_nominal + 0.5 * u_samples  # add exploration noise

    costs, trajectories = jax.vmap(rollout_trajectory, in_axes=(0, 0))(u_samples, random.split(key, num_samples))
    weights = jnp.exp(-costs / lambda_)
    weights /= jnp.sum(weights)

    u_opt = jnp.sum(weights[:, None, None] * u_samples, axis=0)

    return u_opt[0]  # return first control to apply


