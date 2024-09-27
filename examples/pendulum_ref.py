from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx

from mujoco_sysid.mjx.convert import logchol2theta, theta2logchol
from mujoco_sysid.mjx.model import create_rollout
from mujoco_sysid.mjx.parameters import get_dynamic_parameters, set_dynamic_parameters


@jax.jit
def parameters_map(parameters: jnp.ndarray, model: mjx.Model) -> mjx.Model:
    """Map new parameters to the model."""
    log_cholesky, damping, friction_loss = jnp.split(parameters, [10, 11])
    inertial_parameters = logchol2theta(log_cholesky)
    model = set_dynamic_parameters(model, 1, inertial_parameters)
    return model.tree_replace(
        {
            "dof_damping": model.dof_damping.at[0].set(damping[0]),
            "dof_frictionloss": model.dof_frictionloss.at[0].set(friction_loss[0]),
        }
    )


rollout_trajectory = jax.jit(create_rollout(parameters_map))


# Initialize random key
key = jax.random.PRNGKey(0)

# Load the model
MJCF_PATH = "../data/models/pendulum/pendulum.xml"
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)
model.opt.integrator = 1

# Setting up constraint solver to ensure differentiability and faster simulations
model.opt.solver = 2  # 2 corresponds to Newton solver
model.opt.iterations = 2
model.opt.ls_iterations = 10

mjx_model = mjx.put_model(model)

# Load test data
TEST_DATA_PATH = "../data/trajectories/pendulum/free_fall_2.csv"
data_array = np.genfromtxt(TEST_DATA_PATH, delimiter=",", skip_header=100, skip_footer=2500)
timespan = data_array[:, 0] - data_array[0, 0]
sampling = np.mean(np.diff(timespan))
angle = data_array[:, 1]
velocity = data_array[:, 2]
control = data_array[:, 3]

model.opt.timestep = sampling

HORIZON = 100
N_INTERVALS = len(timespan) // HORIZON - 1
timespan = timespan[: N_INTERVALS * HORIZON]
angle = angle[: N_INTERVALS * HORIZON]
velocity = velocity[: N_INTERVALS * HORIZON]
control = control[: N_INTERVALS * HORIZON]

# Prepare data for simulation and optimization
initial_state = jnp.array([angle[0], velocity[0]])
true_trajectory = jnp.column_stack((angle, velocity))
control_inputs = jnp.array(control)

interval_true_trajectory = true_trajectory[::HORIZON]
interval_controls = control_inputs.reshape(N_INTERVALS, HORIZON)

# Get default parameters from the model
default_parameters = jnp.concatenate(
    [theta2logchol(get_dynamic_parameters(mjx_model, 1)), mjx_model.dof_damping, mjx_model.dof_frictionloss]
)

# //////////////////////////////////////
# SIMULATION BATCHES: THIS WILL BE HANDY IN OPTIMIZATION

# Vectorize over both initial states and control inputs
batched_rollout = jax.jit(jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0)))

# Create a batch of initial states
key, subkey = jax.random.split(key)
batch_initial_states = jax.random.uniform(subkey, (N_INTERVALS, 2), minval=-0.1, maxval=0.1) + initial_state
# Create a batch of control input sequences
key, subkey = jax.random.split(key)
batch_control_inputs = jax.random.normal(subkey, (N_INTERVALS, HORIZON)) * 0.1  # + control_inputs
# Run warm up for batched rollout
t1 = perf_counter()
batched_trajectories = batched_rollout(default_parameters, mjx_model, batch_initial_states, batch_control_inputs)
t2 = perf_counter()
print(f"Batch simulation time: {t2 - t1} seconds")

# Run batched rollout on shor horizon data from pendulum
interval_initial_states = true_trajectory[::HORIZON]
interval_controls = control_inputs.reshape(N_INTERVALS, HORIZON)
t1 = perf_counter()
batched_states_trajectories = batched_rollout(
    default_parameters * 0.7, mjx_model, interval_initial_states, interval_controls
)
t2 = perf_counter()
print(f"Batch simulation time: {t2 - t1} seconds")

batched_states_trajectories = np.array(batched_states_trajectories).reshape(N_INTERVALS * HORIZON, 2)

# Plotting simulation results for batсhed state trajectories
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(timespan, angle, label="Actual Angle", color="black", linestyle="dashed", linewidth=2)
plt.plot(timespan, batched_states_trajectories[:, 0], alpha=0.5, color="blue", label="Simulated Angle")
plt.ylabel("Angle (rad)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.legend()
plt.title("Pendulum Dynamics - Bathed State Trajectories")

plt.subplot(2, 2, 3)
plt.plot(timespan, velocity, label="Actual Velocity", color="black", linestyle="dashed", linewidth=2)
plt.plot(timespan, batched_states_trajectories[:, 1], alpha=0.5, color="blue", label="Simulated Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.legend()

# Add phase portrait
plt.subplot(1, 2, 2)
plt.plot(angle, velocity, label="Actual", color="black", linestyle="dashed", linewidth=2)
plt.plot(
    batched_states_trajectories[:, 0], batched_states_trajectories[:, 1], alpha=0.5, color="blue", label="Simulated"
)
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Phase Portrait")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()

# # //////////////////////////////////////////////////
# # PARAMETRIC BATCHES
# # Create a batch of 200 randomized parameters
# num_batches = 200
# key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

# default_log_cholesky_first = default_parameters[0]
# default_damping = default_parameters[-2]
# default_dry_friction = default_parameters[-1]

# randomized_log_cholesky_first = jax.random.uniform(
#     subkey1, (num_batches,), minval=default_log_cholesky_first * 0.8, maxval=default_log_cholesky_first * 1.1
# )

# randomized_damping = jax.random.uniform(
#     subkey2, (num_batches,), minval=default_damping * 0.9, maxval=default_damping * 1.5
# )

# randomized_dry_friction = jax.random.uniform(
#     subkey3, (num_batches,), minval=default_dry_friction * 0.9, maxval=default_dry_friction * 1.5
# )

# # Create a batch of parameters with randomized first log-Cholesky parameter, damping, and dry frictions
# batch_parameters = jnp.tile(default_parameters, (num_batches, 1))
# batch_parameters = batch_parameters.at[:, 0].set(randomized_log_cholesky_first)
# batch_parameters = batch_parameters.at[:, -2].set(randomized_damping)
# batch_parameters = batch_parameters.at[:, -1].set(randomized_dry_friction)


# # Define a batched version of rollout_trajectory using vmap
# batched_parameters_rollout = jax.jit(jax.vmap(rollout_trajectory, in_axes=(0, None, None, None)))

# # Simulation with XML parameters
# xml_trajectory = rollout_trajectory(default_parameters, mjx_model, initial_state, control_inputs)

# # Simulate trajectories with randomized parameters using vmap
# t1 = perf_counter()
# randomized_trajectories = batched_parameters_rollout(batch_parameters, mjx_model, initial_state, control_inputs)
# t2 = perf_counter()

# print(f"Simulation with randomized parameters using vmap took {t2-t1:.2f} seconds.")
# # Plotting simulation results (XML vs Randomized)
# plt.figure(figsize=(10, 5))

# plt.subplot(2, 2, 1)
# plt.plot(timespan, angle, label="Actual Angle", color="black", linestyle="dashed", linewidth=2)
# for trajectory in randomized_trajectories:
#     plt.plot(timespan, trajectory[:, 0], alpha=0.02, color="blue")
# plt.plot(timespan, xml_trajectory[:, 0], label="XML Model Angle", color="red", linewidth=2)
# plt.ylabel("Angle (rad)")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.grid(True)
# plt.legend()
# plt.title("Pendulum Dynamics - Randomized Parameters")

# plt.subplot(2, 2, 3)
# plt.plot(timespan, velocity, label="Actual Velocity", color="black", linestyle="dashed", linewidth=2)
# for trajectory in randomized_trajectories:
#     plt.plot(timespan, trajectory[:, 1], alpha=0.02, color="blue")
# plt.plot(timespan, xml_trajectory[:, 1], label="XML Model Velocity", color="red", linewidth=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (rad/s)")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.grid(True)
# plt.legend()

# # Add phase portrait
# plt.subplot(1, 2, 2)
# plt.plot(angle, velocity, label="Actual", color="black", linestyle="dashed", linewidth=2)
# for trajectory in randomized_trajectories:
#     plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.02, color="blue")
# plt.plot(xml_trajectory[:, 0], xml_trajectory[:, 1], label="XML Model", color="red", linewidth=2)
# plt.xlabel("Angle (rad)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Phase Portrait")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()


# randomized_log_cholesky_first = jax.random.uniform(
#     subkey1, (num_batches,), minval=default_log_cholesky_first * 0.8, maxval=default_log_cholesky_first * 1.1
# )

# randomized_damping = jax.random.uniform(
#     subkey2, (num_batches,), minval=default_damping * 0.9, maxval=default_damping * 1.5
# )

# randomized_dry_friction = jax.random.uniform(
#     subkey3, (num_batches,), minval=default_dry_friction * 0.9, maxval=default_dry_friction * 1.5
# )

# # Create a batch of parameters with randomized first log-Cholesky parameter, damping, and dry frictions
# batch_parameters = jnp.tile(default_parameters, (num_batches, 1))
# batch_parameters = batch_parameters.at[:, 0].set(randomized_log_cholesky_first)
# batch_parameters = batch_parameters.at[:, -2].set(randomized_damping)
# batch_parameters = batch_parameters.at[:, -1].set(randomized_dry_friction)

# # Simulate trajectories with randomized parameters using vmap
# t1 = perf_counter()
# randomized_trajectories = batched_parameters_rollout(batch_parameters, mjx_model, initial_state, control_inputs)
# t2 = perf_counter()
# print(f"Simulation with randomized parameters using vmap took {t2-t1:.2f} seconds.")
