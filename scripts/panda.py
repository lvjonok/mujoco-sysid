import mujoco
import mujoco.viewer
import numpy as np
import time
from mujoco_logger import SimLogger
from utils import update_actuator, ActuatorMotor

from robot_descriptions.panda_mj_description import MJCF_PATH


# # change the actual name of the file to scene.xml
# MJCF_PATH = MJCF_PATH[: -len("panda.xml")]
# MJCF_PATH += "scene.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

for actuator_id in range(model.nu):
    actuator = ActuatorMotor()
    update_actuator(model, actuator_id, actuator)

# find limits except the last two fingers
lower, upper = np.zeros(model.nq - 2), np.zeros(model.nq - 2)
for jnt_idx in range(model.nq - 2):
    lower[jnt_idx] = model.joint(jnt_idx).range[0]
    upper[jnt_idx] = model.joint(jnt_idx).range[1]

np.random.seed(0)

# each 2 seconds we want to change the target position
phase_time = 1.0
phases = 0
kp = 20
kd = 10
target_q = np.zeros(model.nq - 2)

with (
    mujoco.viewer.launch_passive(model, data) as viewer,
    SimLogger(model, data, output_filepath="data/panda_data.json") as logger,
):
    while viewer.is_running() and data.time < 10:
        step_start = time.time()

        if data.time > phase_time * phases:
            target_q = np.random.uniform(lower, upper)
            phases += 1

        # PD controller
        target_acc = kp * (target_q - data.qpos[:-2]) + kd * (
            np.zeros(model.nv - 2) - data.qvel[:-2]
        )

        # solve inverse dynamics
        prev = data.qacc[:-2]
        data.qacc[:-2] = target_acc
        mujoco.mj_inverse(model, data)
        sol = data.qfrc_inverse

        data.ctrl[:-1] = sol[:-2]
        data.qacc[:-2] = prev

        mujoco.mj_step(model, data)
        logger.record()

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
