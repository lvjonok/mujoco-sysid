import jax
import jax.numpy as jnp
from jax import jit, vmap
import mujoco
from mujoco import mjx
import numpy as np
from time import perf_counter
from mujoco_sysid.mjx.convert import logchol2theta, theta2logchol
from mujoco_sysid.mjx.parameters import get_dynamic_parameters, set_dynamic_parameters

# from robot_descriptions.h1_mj_description import MJCF_PATH
from robot_descriptions.iiwa14_mj_description import MJCF_PATH

# Load the model and data
mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)

# Turn off collisions
for geom_id in range(mj_model.ngeom):
    mj_model.geom_contype[geom_id] = 0
    mj_model.geom_conaffinity[geom_id] = 0

mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
BATCH_DIM = 1000

# Get true parameters
true_parameters = get_dynamic_parameters(mj_model, 1)
true_logchol_parameters = theta2logchol(true_parameters)


@jax.jit
def parametric_step(state, control, logchol_parameters):
    # update the parameters
    parameters = logchol2theta(logchol_parameters)
    new_model = set_dynamic_parameters(mjx_model, 1, parameters)

    mjx_data = mjx.make_data(new_model)
    mjx_data = mjx_data.replace(qpos=state[: mjx_model.nq], qvel=state[mjx_model.nq :], ctrl=control)
    mjx_data = mjx.step(new_model, mjx_data)
    state = jnp.hstack((mjx_data.qpos, mjx_data.qvel))
    return state


# Batch version of the parametric_step function
batched_step = jit(vmap(parametric_step, in_axes=(0, 0, 0)))

# Generate random states, controls, and parameters for BATCH_DIM
np.random.seed(0)
states = np.random.randn(BATCH_DIM, mj_model.nq + mj_model.nv)
controls = np.random.randn(BATCH_DIM, mj_model.nu)

# Generate parameters with small random deviations
parameter_deviation = 0.01  # 1% deviation
logchol_parameters = np.array([true_logchol_parameters for _ in range(BATCH_DIM)])
logchol_parameters += np.random.randn(*logchol_parameters.shape) * parameter_deviation * np.abs(logchol_parameters)

# Measure JIT compilation time (First call)
print("Starting the JIT compilation process...")
start_jit = perf_counter()
batched_step(states, controls, logchol_parameters)
end_jit = perf_counter()
jit_compilation_time = end_jit - start_jit
print(f"Time taken for JIT compilation: {jit_compilation_time * 1000:.3f} ms")

# Warmup call
print("Measuring the first warmup call time...")
start_warmup = perf_counter()
batched_step(states, controls, logchol_parameters)
end_warmup = perf_counter()
warmup_time = end_warmup - start_warmup
print(f"Time taken for the first warmup call: {warmup_time * 1000:.3f} ms")

# Measure the average runtime per cycle over 100 iterations
num_iterations = 100
total_runtime = 0

print("Starting the runtime per cycle measurement loop...")
for i in range(num_iterations):
    states = np.random.randn(BATCH_DIM, mj_model.nq + mj_model.nv)
    controls = np.random.randn(BATCH_DIM, mj_model.nu)
    logchol_parameters = np.array([true_logchol_parameters for _ in range(BATCH_DIM)])
    logchol_parameters += np.random.randn(*logchol_parameters.shape) * parameter_deviation * np.abs(logchol_parameters)

    t_start = perf_counter()
    batched_step(states, controls, logchol_parameters)
    t_end = perf_counter()
    runtime_per_cycle = t_end - t_start
    total_runtime += runtime_per_cycle

# Calculate average runtime per cycle
average_runtime_per_cycle = total_runtime / num_iterations
print(f"Average runtime per cycle: {average_runtime_per_cycle * 1000:.3f} ms")
