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
import os
import optax
from mujoco.mjx._src.types import IntegratorType

# SHOULD WE MOVE THIS IN TO MODULE INIT?
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


@jax.jit
def parameters_map(parameters: jnp.ndarray, model: mjx.Model) -> mjx.Model:
    """Map new parameters to the model."""
    log_cholesky, log_damping, log_friction_loss = jnp.split(parameters, [10, 11])
    inertial_parameters = logchol2theta(log_cholesky)
    model = set_dynamic_parameters(model, 0, inertial_parameters)
    damping = jnp.exp(log_damping[0])
    friction_loss = jnp.exp(log_friction_loss[0])
    return model.tree_replace(
        {
            "dof_damping": model.dof_damping.at[0].set(damping),
            "dof_frictionloss": model.dof_frictionloss.at[0].set(friction_loss),
        }
    )


rollout_trajectory = jax.jit(create_rollout(parameters_map))


# Initialize random key
key = jax.random.PRNGKey(0)

# Load the model
MJCF_PATH = "../../data/models/pendulum/pendulum.xml"
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)
model.opt.integrator = IntegratorType.EULER

# Setting up constraint solver to ensure differentiability and faster simulations
model.opt.solver = 2  # 2 corresponds to Newton solver
model.opt.iterations = 1
model.opt.ls_iterations = 10

mjx_model = mjx.put_model(model)

# Load test data
TEST_DATA_PATH = "../../data/trajectories/pendulum/free_fall_2.csv"
data_array = np.genfromtxt(
    TEST_DATA_PATH, delimiter=",", skip_header=100, skip_footer=2500)
timespan = data_array[:, 0] - data_array[0, 0]
sampling = np.mean(np.diff(timespan))
angle = data_array[:, 1]
velocity = data_array[:, 2]
control = data_array[:, 3]

model.opt.timestep = sampling

HORIZON = 10
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
    [theta2logchol(get_dynamic_parameters(mjx_model, 1)),
     jnp.log(mjx_model.dof_damping), jnp.log(mjx_model.dof_frictionloss)]
)

# print()
@jax.jit
def rollout_errors(parameters, states, controls):
    interval_initial_states = states[::HORIZON]
    interval_terminal_states = states[HORIZON+1:][::HORIZON]
    interval_controls = jnp.reshape(controls, (N_INTERVALS, HORIZON))
    batched_rollout = jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))
    batched_states_trajectories = batched_rollout(parameters, mjx_model, interval_initial_states, interval_controls)
    predicted_terminal_points = batched_states_trajectories[:,-1,:]
    loss = jnp.mean(optax.l2_loss(predicted_terminal_points[:-1], interval_terminal_states)) + 0.05*jnp.mean(optax.huber_loss(parameters, jnp.zeros_like(parameters)))
    return loss

start_learning_rate = 1e-3
optimizer = optax.adam(learning_rate = start_learning_rate)


# Initialize parameters of the model + optimizer.
params = jnp.array(0.5*default_parameters)
opt_state = optimizer.init(params)
val_and_grad = jax.jit(jax.value_and_grad(rollout_errors))
loss_val, loss_grad = val_and_grad(params, true_trajectory, control_inputs)

# A simple update loop.
for _ in range(100):
  loss_val, loss_grad = val_and_grad(params, true_trajectory, control_inputs)
  updates, opt_state = optimizer.update(loss_grad, opt_state)
  params = optax.apply_updates(params, updates)
  print(loss_val, params)

# assert jnp.allclose(params, target_params), \
# 'Optimization should retrive the target params used to generate the data.'
# # //////////////////////////////////////
# # SIMULATION BATCHES: THIS WILL BE HANDY IN OPTIMIZATION

# # Vectorize over both initial states and control inputs


# # Create a batch of initial states
# key, subkey = jax.random.split(key)
# batch_initial_states = jax.random.uniform(
#     subkey, (N_INTERVALS, 2), minval=-0.1, maxval=0.1) + initial_state
# # Create a batch of control input sequences
# key, subkey = jax.random.split(key)
# batch_control_inputs = jax.random.normal(
#     subkey, (N_INTERVALS, HORIZON)) * 0.1  # + control_inputs
# # Run warm up for batched rollout
# t1 = perf_counter()
# batched_trajectories = batched_rollout(
#     default_parameters, mjx_model, batch_initial_states, batch_control_inputs)
# t2 = perf_counter()
# print(f"Batch simulation time: {t2 - t1} seconds")

# # Run batched rollout on shor horizon data from pendulum
# interval_initial_states = true_trajectory[::HORIZON]
# interval_terminal_states = true_trajectory[HORIZON+1:][::HORIZON]
# interval_controls = control_inputs.reshape(N_INTERVALS, HORIZON)
# batched_states_trajectories = batched_rollout(
#     default_parameters*0.999999, mjx_model, interval_initial_states, interval_controls)
# t1 = perf_counter()
# t2 = perf_counter()
# print(f"Batch simulation time: {t2 - t1} seconds")

# predicted_terminal_points = np.array(batched_states_trajectories)[:,-1,:]
# batched_states_trajectories = np.array(
#     batched_states_trajectories).reshape(N_INTERVALS * HORIZON, 2)
# # Plotting simulation results for batсhed state trajectories
# plt.figure(figsize=(10, 5))

# plt.subplot(2, 2, 1)
# plt.plot(timespan, angle, label="Actual Angle",
#          color="black", linestyle="dashed", linewidth=2)
# plt.plot(timespan, batched_states_trajectories[:, 0],
#          alpha=0.5, color="blue", label="Simulated Angle")
# plt.plot(timespan, angle, label="Actual Angle",
#          color="black", linestyle="dashed", linewidth=2)
# plt.plot(timespan[HORIZON+1:][::HORIZON], predicted_terminal_points[:-1,0], 'ob')
# plt.plot(timespan[HORIZON+1:][::HORIZON], interval_terminal_states[:, 0], 'or')
# plt.ylabel("Angle (rad)")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.legend()
# plt.title("Pendulum Dynamics - Bathed State Trajectories")

# plt.subplot(2, 2, 3)
# plt.plot(timespan, velocity, label="Actual Velocity",
#          color="black", linestyle="dashed", linewidth=2)
# plt.plot(timespan[HORIZON+1:][::HORIZON], predicted_terminal_points[:-1,1], 'ob')
# plt.plot(timespan[HORIZON+1:][::HORIZON], interval_terminal_states[:, 1], 'or')
# plt.plot(timespan, batched_states_trajectories[:, 1],
#          alpha=0.5, color="blue", label="Simulated Velocity")
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (rad/s)")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.legend()

# # Add phase portrait
# plt.subplot(1, 2, 2)
# plt.plot(angle, velocity, label="Actual",
#          color="black", linestyle="dashed", linewidth=2)
# plt.plot(
#     batched_states_trajectories[:, 0], batched_states_trajectories[:, 1], alpha=0.5, color="blue", label="Simulated"
# )
# plt.plot(predicted_terminal_points[:-1,0], predicted_terminal_points[:-1,1], 'ob')
# plt.plot(interval_terminal_states[:, 0], interval_terminal_states[:, 1], 'or')
# plt.xlabel("Angle (rad)")
# plt.ylabel("Angular Velocity (rad/s)")
# plt.title("Phase Portrait")
# plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
# plt.legend()

# plt.tight_layout()
# plt.show()
# # TODO:
# # Optimization

