from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_sysid.mjx.model import create_rollout
from mujoco_sysid.utils import mjx2mujoco
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
    log_mass, log_damping, log_friction_loss = parameters[0], parameters[1], parameters[2]
    # inertial_parameters = logchol2theta(log_cholesky)
    # model = set_dynamic_parameters(model, 1, inertial_parameters)
    mass = jnp.exp(log_mass)
    damping = jnp.exp(log_damping)
    friction_loss = jnp.exp(log_friction_loss)
    return model.tree_replace(
        {
            "body_mass": model.body_mass.at[1].set(mass),
            "dof_damping": model.dof_damping.at[0].set(damping),
            "dof_frictionloss": model.dof_frictionloss.at[0].set(friction_loss),
        }
    )


rollout_trajectory = jax.jit(create_rollout(parameters_map))


# Initialize random key
key = jax.random.PRNGKey(0)

# Load the model
MJCF_PATH = "models/pendulum.xml"

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)
model.opt.integrator = IntegratorType.EULER

# Setting up constraint solver to ensure differentiability and faster simulations
model.opt.solver = 2  # 2 corresponds to Newton solver
model.opt.iterations = 1
model.opt.ls_iterations = 10


mjx_model = mjx.put_model(model)

# Load test data
LEARNING_DATA_PATH = "data/free_fall_2.csv"
data_array = np.genfromtxt(LEARNING_DATA_PATH, delimiter=",", skip_header=100, skip_footer=1000)
timespan = data_array[:, 0] - data_array[0, 0]
sampling = np.mean(np.diff(timespan))
angle = data_array[:, 1]
velocity = data_array[:, 2]
control = data_array[:, 3]

model.opt.timestep = sampling

HORIZON = 40
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
    [jnp.log(jnp.array([mjx_model.body_mass[1]])), jnp.log(mjx_model.dof_damping), jnp.log(mjx_model.dof_frictionloss)]
)


@jax.jit
def rollout_errors(parameters, states, controls):
    # TODO: Use the full trajecttory in shouting not only las point
    interval_initial_states = states[::HORIZON]
    interval_terminal_states = states[HORIZON + 1 :][::HORIZON]
    interval_controls = jnp.reshape(controls, (N_INTERVALS, HORIZON))
    batched_rollout = jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))
    batched_states_trajectories = batched_rollout(parameters, mjx_model, interval_initial_states, interval_controls)
    predicted_terminal_points = batched_states_trajectories[:, -1, :]
    loss = jnp.mean(
        optax.l2_loss(predicted_terminal_points[:-1], interval_terminal_states)
    )  # + 0.05*jnp.mean(optax.huber_loss(parameters, jnp.zeros_like(parameters)))
    return loss


@jax.jit
def rollout_errors(parameters, states, controls):
    # TODO: Use the full trajecttory in shouting not only las point
    interval_initial_states = states[::HORIZON]
    interval_terminal_states = states[HORIZON + 1 :][::HORIZON]
    interval_controls = jnp.reshape(controls, (N_INTERVALS, HORIZON))
    batched_rollout = jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))
    batched_states_trajectories = batched_rollout(parameters, mjx_model, interval_initial_states, interval_controls)
    predicted_terminal_points = batched_states_trajectories[:, -1, :]
    loss = jnp.mean(
        optax.l2_loss(predicted_terminal_points[:-1], interval_terminal_states)
    )  # + 0.05*jnp.mean(optax.huber_loss(parameters, jnp.zeros_like(parameters)))
    return loss


optimizer = optax.adam(learning_rate=0.6)


# Initialize parameters of the model + optimizer.
estimated_parameters = jnp.array(default_parameters)
opt_state = optimizer.init(estimated_parameters)
val_and_grad = jax.jit(jax.value_and_grad(rollout_errors))
loss_val, loss_grad = val_and_grad(estimated_parameters, true_trajectory, control_inputs)
# Define thresholds for early stopping
COST_THRESHOLD = 1e-7
PARAM_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
# Initialize variables to store previous values
prev_loss_val = float("inf")
prev_estimated_parameters = estimated_parameters

# A simple update loop with early stopping
for iteration in range(MAX_ITERATIONS):
    loss_val, loss_grad = val_and_grad(estimated_parameters, true_trajectory, control_inputs)
    updates, opt_state = optimizer.update(loss_grad, opt_state)
    estimated_parameters = optax.apply_updates(estimated_parameters, updates)

    # Calculate cost and parameter increments
    cost_increment = abs(prev_loss_val - loss_val)
    param_increment = jnp.max(jnp.abs(estimated_parameters - prev_estimated_parameters))

    if iteration % 10 == 0:
        print("Loss at iteration", iteration, ":  ", loss_val)
        print("Params at iteration", iteration, ":  ", estimated_parameters)
        print("Gradients at iteration", iteration, ":  ", loss_grad)
        print("Cost increment:", cost_increment)
        print("Parameter increment:", param_increment)
        print("---")

    # Check for early stopping conditions
    if cost_increment < COST_THRESHOLD and param_increment < PARAM_THRESHOLD:
        print(f"Optimization converged at iteration {iteration}")
        print(f"Final loss: {loss_val}")
        print(f"Final parameters: {estimated_parameters}")
        break

    # Update previous values
    prev_loss_val = loss_val
    prev_estimated_parameters = estimated_parameters

# If the loop completes without breaking, print the final results
else:
    print("Optimization completed without convergence")
    print(f"Final loss: {loss_val}")
    print(f"Final parameters: {estimated_parameters}")

# Now lets perform rollouts on full trajectory to ensure that new model is better
TEST_DATA_PATH = "data/free_fall_1.csv"
data_array = np.genfromtxt(TEST_DATA_PATH, delimiter=",", skip_header=10, skip_footer=3000)
timespan = data_array[:, 0] - data_array[0, 0]
sampling = np.mean(np.diff(timespan))
angle = data_array[:, 1]
velocity = data_array[:, 2]
control = jnp.array(data_array[:, 3])
initial_state = jnp.array([angle[0], velocity[0]])

old_rollout = rollout_trajectory(default_parameters, mjx_model, initial_state, control)
new_rollout = rollout_trajectory(estimated_parameters, mjx_model, initial_state, control)


# Plotting simulation results
plt.figure(figsize=(10, 5))
# Angle plot
plt.subplot(2, 2, 1)
plt.plot(timespan, angle, label="Actual Angle", color="black", linestyle="dashed", linewidth=2)
plt.plot(timespan, old_rollout[:, 0], color="blue", alpha=0.3, label="Default Model")
plt.plot(timespan, new_rollout[:, 0], color="red", label="Optimized Model")
plt.ylabel("Angle (rad)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)


# Velocity plot
plt.subplot(2, 2, 3)
plt.plot(timespan, velocity, label="Actual Velocity", color="black", linestyle="dashed", linewidth=2)
plt.plot(timespan, old_rollout[:, 1], color="blue", alpha=0.3, label="Default Model")
plt.plot(timespan, new_rollout[:, 1], color="red", label="Optimized Model")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)


# Phase portrait
plt.subplot(1, 2, 2)
plt.plot(angle, velocity, label="Actual", color="black", linestyle="dashed", linewidth=2)
plt.plot(old_rollout[:, 0], old_rollout[:, 1], color="blue", alpha=0.3, label="Default Model")
plt.plot(new_rollout[:, 0], new_rollout[:, 1], color="red", label="Optimized Model")
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Phase Portrait")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig("plots/learning_results.png", dpi=300)  # save the figure as a PNG file
plt.show()

# Print the default and estimated parameters
print("Default parameters (mass, damping, friction):", np.exp(default_parameters))
print("Estimated parameters (mass, damping, friction):", np.exp(estimated_parameters))

# Calculate and print the mean squared error for both models
mse_default_angle = np.mean((angle - old_rollout[:, 0]) ** 2)
mse_default_velocity = np.mean((velocity - old_rollout[:, 1]) ** 2)
mse_optimized_angle = np.mean((angle - new_rollout[:, 0]) ** 2)
mse_optimized_velocity = np.mean((velocity - new_rollout[:, 1]) ** 2)

print(f"Default model MSE - Angle: {mse_default_angle:.6f}, Velocity: {mse_default_velocity:.6f}")
print(f"Optimized model MSE - Angle: {mse_optimized_angle:.6f}, Velocity: {mse_optimized_velocity:.6f}")

# Wa may also save the model to mujoco format for further simulation
# get updated MJX model
updated_mjx_model = parameters_map(estimated_parameters, mjx_model)
updated_mj_model = mjx2mujoco(model, updated_mjx_model)

# Dont know how to properly save the updated model( Mass is not changing, maybe use spec
# print(updated_mj_model.body_mass)
# mujoco.mj_saveLastXML("models/pendulum_estimated.xml", updated_mj_model)
