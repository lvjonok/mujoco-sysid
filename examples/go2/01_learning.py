import os

import jax
import jax.numpy as jnp
import mujoco
import optax
from jax.typing import ArrayLike
from mujoco import mjx
from mujoco.mjx._src.types import IntegratorType
from mujoco_logger import SimLog
from robot_descriptions.go2_mj_description import MJCF_PATH

from mujoco_sysid.mjx.model import create_rollout

# update path for mjx model
MJCF_PATH = MJCF_PATH[: -len("go2.xml")] + "scene_mjx.xml"

# SHOULD WE MOVE THIS IN TO MODULE INIT?
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

# Initialize random key
key = jax.random.PRNGKey(0)

# Load the model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)
model.opt.integrator = IntegratorType.EULER

# Setting up constraint solver to ensure differentiability and faster simulations
model.opt.solver = 2  # 2 corresponds to Newton solver
model.opt.iterations = 1
model.opt.ls_iterations = 10

mjx_model = mjx.put_model(model)

default_parameters = jnp.concatenate(
    [
        jnp.log(jnp.array([mjx_model.body_mass[1]])),
        jnp.log(mjx_model.dof_damping[6:]),
        jnp.log(mjx_model.dof_frictionloss[6:] + 0.2),
    ]
)
print(f"Default parameters: {default_parameters}")

@jax.jit
def parameters_map(parameters: ArrayLike, model: mjx.Model) -> mjx.Model:
    """Map new parameters to the model."""
    log_mass, log_damping, log_friction = parameters[0], parameters[1:13], parameters[13:]

    mass = jnp.exp(log_mass)
    damping = jnp.exp(log_damping)
    friction = jnp.exp(log_friction)
    return model.tree_replace(
        {
            "body_mass": model.body_mass.at[1].set(mass),
            "dof_damping": model.dof_damping.at[6:].set(damping),
            "dof_frictionloss": model.dof_frictionloss.at[6:].set(friction),
        }
    )


rollout_trajectory = create_rollout(parameters_map)
batched_rollout = jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))


# open log
log = SimLog("data/go2_twist_active.json")
states = jnp.column_stack([log.data("qpos"), log.data("qvel")])
controls = jnp.array(log.u)

HORIZON = 20
N_INTERVALS = len(log) // HORIZON - 1
print(N_INTERVALS)
N_INTERVALS = 50  # reduce the number of intervals for faster computation
states = states[: N_INTERVALS * HORIZON]
controls = controls[: N_INTERVALS * HORIZON]


@jax.jit
def rollout_errors(parameters, states, controls):
    initial_states = states[::HORIZON]  # each interval starts with the initial state
    terminal_states = states[HORIZON + 1 :: HORIZON]  # each interval ends with the final state
    controls0 = controls.reshape(N_INTERVALS, HORIZON, -1)

    simulated = batched_rollout(parameters, mjx_model, initial_states, controls0)
    predicted_terminal_points = simulated[:, -1, :]

    return jnp.mean(optax.l2_loss(predicted_terminal_points[:-1], terminal_states))


optimizer = optax.adam(learning_rate=0.5)

# Initialize parameters of the model + optimizer.
estimated_parameters = jnp.array(default_parameters) + 0.4 * jax.random.normal(key, default_parameters.shape)
opt_state = optimizer.init(estimated_parameters)
val_and_grad = jax.jit(jax.value_and_grad(rollout_errors))
loss_val, loss_grad = val_and_grad(estimated_parameters, states, controls)

print("Initial loss: ", loss_val)

# Define thresholds for early stopping
COST_THRESHOLD = 1e-7
PARAM_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
# Initialize variables to store previous values
prev_loss_val = float("inf")
prev_estimated_parameters = estimated_parameters

# A simple update loop with early stopping
for iteration in range(MAX_ITERATIONS):
    loss_val, loss_grad = val_and_grad(estimated_parameters, states, controls)
    updates, opt_state = optimizer.update(loss_grad, opt_state)
    estimated_parameters = optax.apply_updates(estimated_parameters, updates)

    # Calculate cost and parameter increments
    cost_increment = abs(prev_loss_val - loss_val)
    param_increment = jnp.max(jnp.abs(estimated_parameters - prev_estimated_parameters))

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
