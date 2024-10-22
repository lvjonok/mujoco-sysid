import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_sysid.mjx.model import create_rollout
from mujoco_sysid.utils import mjx2mujoco
import os
import optax
from mujoco.mjx._src.types import IntegratorType
import mediapy as media
from _plotting_utils import plot_simulation_errors


# SHOULD WE MOVE THIS IN TO MODULE INIT?
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


@jax.jit
def parameters_map(parameters: jnp.ndarray, model: mjx.Model) -> mjx.Model:
    """Map new parameters to the model."""
    log_mass, log_damping, log_friction_loss = parameters[0], parameters[1], parameters[2]
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


def load_data(file_path, skip_footer=0):
    data_array = np.genfromtxt(file_path, delimiter=",", skip_header=10, skip_footer=skip_footer)
    timespan = data_array[:, 0] - data_array[0, 0]
    sampling = np.mean(np.diff(timespan))
    angle = data_array[:, 1]
    velocity = data_array[:, 2]
    control = data_array[:, 3]
    return timespan, sampling, angle, velocity, control


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

# Load learning data
LEARNING_DATA_PATH = "data/harmonic_input_1.csv"
timespan, sampling, angle, velocity, control = load_data(LEARNING_DATA_PATH, skip_footer=500)

model.opt.timestep = sampling

HORIZON = 50
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
def batched_rollout(parameters, model, initial_states, controls):
    return jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))(parameters, model, initial_states, controls)

@jax.jit
def loss_and_rollouts(parameters, states, controls):
    interval_initial_states = states[::HORIZON]
    interval_terminal_states = states[HORIZON + 1:][::HORIZON]
    interval_controls = jnp.reshape(controls, (N_INTERVALS, HORIZON))
    batched_states_trajectories = batched_rollout(parameters, mjx_model, interval_initial_states, interval_controls)
    predicted_terminal_points = batched_states_trajectories[:, -1, :]
    loss = jnp.mean(optax.l2_loss(predicted_terminal_points[:-1], interval_terminal_states))
    return loss, (batched_states_trajectories, predicted_terminal_points)

optimizer = optax.adam(learning_rate=0.5)


# Initialize parameters of the model + optimizer.
estimated_parameters = jnp.array(default_parameters)
opt_state = optimizer.init(estimated_parameters)
val_and_grad = jax.jit(jax.value_and_grad(loss_and_rollouts, has_aux=True))

# Define thresholds for early stopping
COST_THRESHOLD = 1e-7
PARAM_THRESHOLD = 1e-6
MAX_ITERATIONS = 100
# Initialize variables to store previous values
prev_loss_val = float("inf")
prev_estimated_parameters = estimated_parameters

# A simple update loop with early stopping
frames = []
for iteration in range(MAX_ITERATIONS):
    (loss_val, (batched_states_trajectories, predicted_terminal_points)), loss_grad = val_and_grad(estimated_parameters, true_trajectory, control_inputs)
    updates, opt_state = optimizer.update(loss_grad, opt_state)
    estimated_parameters = optax.apply_updates(estimated_parameters, updates)

    # Calculate cost and parameter increments
    cost_increment = abs(prev_loss_val - loss_val)
    param_increment = jnp.max(jnp.abs(estimated_parameters - prev_estimated_parameters))

    if iteration % 2 == 0:
        interval_terminal_states = true_trajectory[HORIZON + 1:][::HORIZON]
        
        # Create animation frame
        frame = plot_simulation_errors(
            timespan, 
            angle, 
            velocity, 
            batched_states_trajectories.reshape(-1, 2), 
            predicted_terminal_points, 
            interval_terminal_states, 
            HORIZON,
            save_path=None,
            show=False,
            title = f"Simulation Errors (iter: {iteration})"
        )
        frames.append(frame)

        print("Loss at iteration", iteration, ":  ", loss_val)
        print("Params at iteration", iteration, ":  ", estimated_parameters)
        print("---")

    # Check for early stopping conditions
    if cost_increment < COST_THRESHOLD and param_increment < PARAM_THRESHOLD:
        print(f"Optimization converged at iteration {iteration}")
        break

    # Update previous values
    prev_loss_val = loss_val
    prev_estimated_parameters = estimated_parameters

# Create the plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Save the animation using mediapy
media.write_video("plots/learning_animation.mp4", frames, fps=5)

# Now let's verify the model on a new dataset
TEST_DATA_PATH = "data/harmonic_input_2.csv"
timespan, sampling, angle, velocity, control = load_data(TEST_DATA_PATH, skip_footer=1500)

# Prepare data for verification
N_INTERVALS = len(timespan) // HORIZON
timespan = timespan[:N_INTERVALS * HORIZON]
angle = angle[:N_INTERVALS * HORIZON]
velocity = velocity[:N_INTERVALS * HORIZON]
control = control[:N_INTERVALS * HORIZON]

true_trajectory = jnp.column_stack((angle, velocity))
interval_initial_states = true_trajectory[::HORIZON]
interval_controls = jnp.array(control).reshape(N_INTERVALS, HORIZON)

# Perform batched rollouts for simulation error plot with estimated parameters
batched_states_trajectories_estimated = batched_rollout(estimated_parameters, mjx_model, interval_initial_states, interval_controls)
predicted_terminal_points_estimated = batched_states_trajectories_estimated[:, -1, :]

# Perform batched rollouts for simulation error plot with default parameters
batched_states_trajectories_default = batched_rollout(default_parameters, mjx_model, interval_initial_states, interval_controls)
predicted_terminal_points_default = batched_states_trajectories_default[:, -1, :]
interval_terminal_states = true_trajectory[HORIZON::HORIZON]

# Plot simulation errors for default parameters
plot_simulation_errors(
    timespan, 
    angle, 
    velocity, 
    batched_states_trajectories_default.reshape(-1, 2), 
    predicted_terminal_points_default, 
    interval_terminal_states, 
    HORIZON,
    save_path="plots/simulation_errors_default.png",
    show=True,
    title="Simulation Errors (Default Parameters)"
)

# Plot simulation errors for estimated parameters
plot_simulation_errors(
    timespan, 
    angle, 
    velocity, 
    batched_states_trajectories_estimated.reshape(-1, 2), 
    predicted_terminal_points_estimated, 
    interval_terminal_states, 
    HORIZON,
    save_path="plots/simulation_errors_estimated.png",
    show=True,
    title="Simulation Errors (Estimated Parameters)"
)
# Print the default and estimated parameters
print("Default parameters (mass, damping, friction):", np.exp(default_parameters))
print("Estimated parameters (mass, damping, friction):", np.exp(estimated_parameters))

# Calculate and print the mean squared error for both models
mse_default = np.mean((interval_terminal_states - predicted_terminal_points_default[:-1]) ** 2)
mse_estimated = np.mean((interval_terminal_states - predicted_terminal_points_estimated[:-1]) ** 2)

print(f"Default model MSE: {mse_default:.6f}")
print(f"Optimized model MSE: {mse_estimated:.6f}")

# We may also save the model to mujoco format for further simulation
# get updated MJX model
updated_mjx_model = parameters_map(estimated_parameters, mjx_model)
updated_mj_model = mjx2mujoco(model, updated_mjx_model)

# Save estimated model
mujoco.mj_saveLastXML("models/pendulum_estimated.xml", updated_mj_model)
