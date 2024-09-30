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
    log_mass, log_damping, log_friction_loss = parameters[0], parameters[1], parameters[2]
    # inertial_parameters = logchol2theta(log_cholesky)
    # model = set_dynamic_parameters(model, 1, inertial_parameters)
    mass = jnp.exp(log_mass)
    damping = jnp.exp(log_damping)
    friction_loss = jnp.exp(log_friction_loss)
    return model.tree_replace(
        {
            "body_mass": mjx_model.body_mass.at[1].set(mass),
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
TEST_DATA_PATH = "../../data/trajectories/pendulum/harmonic_input_1.csv"
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
    [jnp.log(jnp.array([mjx_model.body_mass[1]])), jnp.log(mjx_model.dof_damping), jnp.log(mjx_model.dof_frictionloss)]
)


# print()
@jax.jit
def rollout_errors(parameters, states, controls):
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


start_learning_rate = 1e-2
optimizer = optax.adam(learning_rate=start_learning_rate)


# Initialize parameters of the model + optimizer.
params = jnp.array(0.7 * default_parameters)
opt_state = optimizer.init(params)
val_and_grad = jax.jit(jax.value_and_grad(rollout_errors))
loss_val, loss_grad = val_and_grad(params, true_trajectory, control_inputs)
print(loss_grad)
# A simple update loop.
for iteration in range(500):
    loss_val, loss_grad = val_and_grad(params, true_trajectory, control_inputs)
    updates, opt_state = optimizer.update(loss_grad, opt_state)
    params = optax.apply_updates(params, updates)
    if iteration % 10 == 0:
        print("Loss at iteration", iteration, ":  ", loss_val)
        print("Params at iteration", iteration, ":  ", params)
        print("Gradients at iteration", iteration, ":  ", loss_grad)

print(jnp.exp(params))
print(jnp.exp(default_parameters))
