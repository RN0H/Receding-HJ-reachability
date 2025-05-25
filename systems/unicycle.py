import jax.numpy as jnp
from hj_reachability import sets
from hj_reachability import dynamics

class Unicycle(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
       
        # Default bounds for control: v ∈ [-1, 1], ω ∈ [-1, 1]
        if control_space is None:
            control_space = sets.Box(
                jnp.array([-1.0, -1.0]),
                jnp.array([1.0, 1.0])
            )

        # Default bounds for disturbance: v ∈ [-0.2, 0.2], ω ∈ [-0.1, 0.1]
        if disturbance_space is None:
            disturbance_space = sets.Box(
                jnp.array([-0.2, -0.1]),
                jnp.array([0.2, 0.1])
            )
        self.control_dim = 2 # [v, omega]
        
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def step(self, state, u, dt=0.1):
        """
        Applies one step of unicycle dynamics: 
            x' = x + v*cos(θ)*dt
            y' = y + v*sin(θ)*dt
            θ' = θ + ω*dt
        """
        x, y, theta = state
        v, omega = u

        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = omega

        next_state = jnp.array([
            x + dx * dt,
            y + dy * dt,
            theta + dtheta * dt
        ])
        return next_state


    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        _, _, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],  # ∂f/∂v and ∂f/∂ω
            [jnp.sin(theta), 0.0],
            [0.0, 1.0]
        ])

    def disturbance_jacobian(self, state, time):
        # Disturbance affects the same channels as control
        _, _, theta = state
        return jnp.array([
            [jnp.cos(theta), 0.0],  # ∂f/∂d_v and ∂f/∂d_ω
            [jnp.sin(theta), 0.0],
            [0.0, 1.0]
        ])
    

if __name__ == "__main__":
    ##### Test #####
    dubinscar = Unicycle()