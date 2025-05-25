import jax.numpy as jnp
import hj_reachability as hj


# --- Patch Extraction ---
def extract_local_patch(grid, values, center_state, patch_shape):
    indices = [jnp.abs(g - s).argmin() for g, s in zip(grid.coordinate_vectors, center_state)]
    
    slices = [slice(max(0, int(i - p // 2)), int(i + p // 2 + 1)) for i, p in zip(indices, patch_shape)]

    local_vectors = [g[s] for g, s in zip(grid.coordinate_vectors, slices)]
    subgrid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(jnp.array([v[0] for v in local_vectors]), jnp.array([v[-1] for v in local_vectors])),
        tuple(len(v) for v in local_vectors),
        periodic_dims=2
    )
    sub_values = values[slices[0], slices[1], slices[2]]

    # For solving a circular patch
    # coords = subgrid.states[..., :2]  # only x,y
    # center_xy = jnp.array([center_state[0], center_state[1]])
    # dists = jnp.linalg.norm(coords - center_xy, axis=-1)
    # sub_values = jnp.where(dists <= 0.3, sub_values, 10)  # ∞: excluded from solve
    return subgrid, sub_values, slices

# --- Reintegrate Function ---
def reintegrate_patch(global_values, patch, slices):
    return global_values.at[slices[0], slices[1], slices[2]].set(patch)


# # --- Receding Horizon control  ---
# def receding_horizon_control(values, grid, dynamics, agent_state, steps=50):
#     for t in range(steps):
#         subgrid, sub_values, slices = extract_local_patch(grid, values, agent_state, (10, 10, 10))
#         updated_sub_values = sub_values
#         current_time = 0.0  # value functions are from time 0
#         for _ in range(num_steps):
#             target_time = current_time - dt  # solve backward in time
#             updated_sub_values = hj.step(
#                 solver_settings,
#                 dynamics,
#                 subgrid,
#                 current_time,
#                 updated_sub_values,
#                 target_time
#             )
#             current_time = target_time  # step back
        
#         values = reintegrate_patch(values, updated_sub_values, slices)
#         values_hist.append(values)
        
#         agent_state = agent_state.at[0].add(control_sequence[t])
#         agent_state = agent_state.at[1].add(control_sequence[t])
#         agent_state = agent_state.at[2].add(control_sequence[t])
#         trajectory.append(agent_state.copy())