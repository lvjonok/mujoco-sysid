from robot_descriptions.go2_mj_description import MJCF_PATH
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
from mujoco_logger import SimLog

# update path for mjx model
MJCF_PATH = MJCF_PATH[: -len("go2.xml")] + "go2_mjx.xml"

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

# open log
log = SimLog("data/go2_twist_active.json")

# query qpos
log.data("qpos")
# query qvel
log.data("qvel")
# query control
log.data("ctrl")
