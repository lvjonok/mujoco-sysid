import jax
import jax.numpy as jnp
import jax.typing as jpt
import mujoco
from mujoco import mjx
import numpy as np
import matplotlib.pyplot as plt
from mujoco_sysid.mjx.convert import logchol2theta, theta2logchol
from mujoco_sysid.mjx.parameters import get_dynamic_parameters, set_dynamic_parameters
from time import perf_counter
import optax


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


@jax.jit
def parametric_step(parameters: jnp.ndarray, model: mjx.Model, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """Perform a step with new parameter mapping."""
    new_model = parameters_map(parameters, model)
    data = mjx.make_data(new_model).replace(qpos=state[: new_model.nq], qvel=state[new_model.nq :], ctrl=control)
    data = mjx.step(new_model, data)
    return jnp.concatenate([data.qpos, data.qvel])


@jax.jit
def rollout_trajectory(
    parameters: jnp.ndarray, model: mjx.Model, initial_state: jnp.ndarray, control_inputs: jnp.ndarray
) -> jnp.ndarray:
    """Rollout a trajectory given parameters, initial state, and control inputs."""

    def step_fn(state, control):
        new_state = parametric_step(parameters, model, state, control)
        return new_state, new_state

    (_, states) = jax.lax.scan(step_fn, initial_state, control_inputs)
    return states


# Prepare data for simulation and optimization
initial_state = jnp.array([angle[0], velocity[0]])
true_trajectory = jnp.column_stack((angle, velocity))
control_inputs = jnp.array(control)

# Get default parameters from the model
default_parameters = jnp.concatenate(
    [theta2logchol(get_dynamic_parameters(mjx_model, 1)), mjx_model.dof_damping, mjx_model.dof_frictionloss]
)

# Simulation with XML parameters
xml_trajectory = rollout_trajectory(default_parameters, mjx_model, initial_state, control_inputs)

# Create a batch of 200 randomized parameters
num_batches = 200
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

default_log_cholesky_first = default_parameters[0]
default_damping = default_parameters[-2]
default_dry_friction = default_parameters[-1]

randomized_log_cholesky_first = jax.random.uniform(
    subkey1, (num_batches,), minval=default_log_cholesky_first * 0.8, maxval=default_log_cholesky_first * 1.1
)

randomized_damping = jax.random.uniform(
    subkey2, (num_batches,), minval=default_damping * 0.9, maxval=default_damping * 1.5
)

randomized_dry_friction = jax.random.uniform(
    subkey3, (num_batches,), minval=default_dry_friction * 0.9, maxval=default_dry_friction * 1.5
)

# Create a batch of parameters with randomized first log-Cholesky parameter, damping, and dry frictions
batch_parameters = jnp.tile(default_parameters, (num_batches, 1))
batch_parameters = batch_parameters.at[:, 0].set(randomized_log_cholesky_first)
batch_parameters = batch_parameters.at[:, -2].set(randomized_damping)
batch_parameters = batch_parameters.at[:, -1].set(randomized_dry_friction)

# Define a batched version of rollout_trajectory using vmap
batched_rollout = jax.jit(jax.vmap(rollout_trajectory, in_axes=(0, None, None, None)))


# Simulate trajectories with randomized parameters using vmap
t1 = perf_counter()
randomized_trajectories = batched_rollout(batch_parameters, mjx_model, initial_state, control_inputs)
t2 = perf_counter()

print(f"Simulation with randomized parameters using vmap took {t2-t1:.2f} seconds.")
# Plotting simulation results (XML vs Randomized)
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(timespan, angle, label="Actual Angle", color="black", linestyle="dashed", linewidth=2)
for trajectory in randomized_trajectories:
    plt.plot(timespan, trajectory[:, 0], alpha=0.02, color="blue")
plt.plot(timespan, xml_trajectory[:, 0], label="XML Model Angle", color="red", linewidth=2)
plt.ylabel("Angle (rad)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.grid(True)
plt.legend()
plt.title("Pendulum Dynamics - Randomized Parameters")

plt.subplot(2, 2, 3)
plt.plot(timespan, velocity, label="Actual Velocity", color="black", linestyle="dashed", linewidth=2)
for trajectory in randomized_trajectories:
    plt.plot(timespan, trajectory[:, 1], alpha=0.02, color="blue")
plt.plot(timespan, xml_trajectory[:, 1], label="XML Model Velocity", color="red", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.grid(True)
plt.legend()

# Add phase portrait
plt.subplot(1, 2, 2)
plt.plot(angle, velocity, label="Actual", color="black", linestyle="dashed", linewidth=2)
for trajectory in randomized_trajectories:
    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.02, color="blue")
plt.plot(xml_trajectory[:, 0], xml_trajectory[:, 1], label="XML Model", color="red", linewidth=2)
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Phase Portrait")
plt.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


randomized_log_cholesky_first = jax.random.uniform(
    subkey1, (num_batches,), minval=default_log_cholesky_first * 0.8, maxval=default_log_cholesky_first * 1.1
)

randomized_damping = jax.random.uniform(
    subkey2, (num_batches,), minval=default_damping * 0.9, maxval=default_damping * 1.5
)

randomized_dry_friction = jax.random.uniform(
    subkey3, (num_batches,), minval=default_dry_friction * 0.9, maxval=default_dry_friction * 1.5
)

# Create a batch of parameters with randomized first log-Cholesky parameter, damping, and dry frictions
batch_parameters = jnp.tile(default_parameters, (num_batches, 1))
batch_parameters = batch_parameters.at[:, 0].set(randomized_log_cholesky_first)
batch_parameters = batch_parameters.at[:, -2].set(randomized_damping)
batch_parameters = batch_parameters.at[:, -1].set(randomized_dry_friction)

# Simulate trajectories with randomized parameters using vmap
t1 = perf_counter()
randomized_trajectories = batched_rollout(batch_parameters, mjx_model, initial_state, control_inputs)
t2 = perf_counter()
print(f"Simulation with randomized parameters using vmap took {t2-t1:.2f} seconds.")

# Optimization

# # Error function
# def trajectory_error(parameters: jnp.ndarray, model: mjx.Model, initial_state: jnp.ndarray, control_inputs: jnp.ndarray, true_trajectory: jnp.ndarray) -> jnp.ndarray:
#     predicted_trajectory = rollout_trajectory(parameters, model, initial_state, control_inputs)
#     return jnp.mean(jnp.square(predicted_trajectory - true_trajectory))

# # Optimization
# @jax.jit
# def update_step(parameters, opt_state, model, initial_state, control_inputs, true_trajectory):
#     loss, grads = jax.value_and_grad(trajectory_error)(parameters, model, initial_state, control_inputs, true_trajectory)
#     updates, opt_state = optimizer.update(grads, opt_state, parameters)
#     parameters = optax.apply_updates(parameters, updates)
#     return parameters, opt_state, loss

# # Initial parameters for optimization (using randomized parameters as starting point)
# initial_parameters = randomized_parameters

# # Optimization setup
# learning_rate = 1e-3
# optimizer = optax.adam(learning_rate)
# opt_state = optimizer.init(initial_parameters)

# # Optimization loop
# num_iterations = 1000
# for i in range(num_iterations):
#     initial_parameters, opt_state, loss = update_step(initial_parameters, opt_state, mjx_model, initial_state, control_inputs, true_trajectory)
#     if i % 100 == 0:
#         print(f"Iteration {i}, Loss: {loss}")

# # Final simulation with learned parameters
# final_trajectory = rollout_trajectory(initial_parameters, mjx_model, initial_state, control_inputs)

# # Plotting optimization results
# plt.figure(figsize=(12, 9))

# plt.subplot(2, 1, 1)
# plt.plot(timespan, angle, label='Actual Angle')
# plt.plot(timespan, xml_trajectory[:, 0], label='XML Model Angle', linestyle='dashed')
# plt.plot(timespan, randomized_trajectory[:, 0], label='Initial Randomized Angle', linestyle='dotted')
# plt.plot(timespan, final_trajectory[:, 0], label='Learned Model Angle', linestyle='dashdot')
# plt.ylabel('Angle (rad)')
# plt.legend()
# plt.title('Pendulum Dynamics Comparison - Optimization')

# plt.subplot(2, 1, 2)
# plt.plot(timespan, velocity, label='Actual Velocity')
# plt.plot(timespan, xml_trajectory[:, 1], label='XML Model Velocity', linestyle='dashed')
# plt.plot(timespan, randomized_trajectory[:, 1], label='Initial Randomized Velocity', linestyle='dotted')
# plt.plot(timespan, final_trajectory[:, 1], label='Learned Model Velocity', linestyle='dashdot')
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (rad/s)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Print learned parameters
# learned_log_cholesky, learned_damping, learned_friction_loss = jnp.split(initial_parameters, [10, 11])
# learned_theta = logchol2theta(learned_log_cholesky)

# print("\nParameters comparison:")
# print("Parameter\t\tXML\t\tRandomized\tLearned")
# print(f"Inertia:\t\t{get_dynamic_parameters(mjx_model, 1)[0]:.6f}\t{logchol2theta(randomized_parameters[:10])[0]:.6f}\t{learned_theta[0]:.6f}")
# print(f"Damping:\t\t{mjx_model.dof_damping[0]:.6f}\t{randomized_parameters[10]:.6f}\t{learned_damping[0]:.6f}")
# print(f"Friction loss:\t{mjx_model.dof_frictionloss[0]:.6f}\t{randomized_parameters[11]:.6f}\t{learned_friction_loss[0]:.6f}")

# TODO: Save the learned parameters to a new XML file
# This would require additional code to modify the XML file with the learned parameters
