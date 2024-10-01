import numpy as np
from quaternion import as_rotation_matrix, quaternion
import mujoco

import copy


def muj2pin(qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray | None = None) -> tuple:
    """
    Converts Mujoco state to Pinocchio state by adjusting the quaternion representation and rotating the velocity.

    This function assumes that the quaternion representation of orientation in the Mujoco state uses a scalar-first
    format (w, x, y, z), while the Pinocchio state uses a scalar-last format (x, y, z, w). It also rotates the linear
    velocity from the world frame to the local frame.

    Args:
        qpos (numpy.ndarray): Mujoco qpos array, which includes position and orientation.
        qvel (numpy.ndarray): Mujoco qvel array, which includes linear and angular velocity.
        qacc (numpy.ndarray, optional): Mujoco qacc array, which includes linear and angular acceleration.

    Returns:
        tuple: A tuple containing two numpy.ndarrays:
            - pin_pos (numpy.ndarray): Pinocchio qpos array, with adjusted quaternion and position.
            - pin_vel (numpy.ndarray): Pinocchio qvel array, with velocity rotated to the local frame.

    """
    # Copy the position and velocity to avoid modifying the original arrays
    pin_pos = qpos.copy()
    pin_vel = qvel.copy()
    pin_acc = qacc.copy() if qacc is not None else None

    if len(pin_pos) == len(qvel):
        if qacc is None:
            return pin_pos, pin_vel

        return pin_pos, pin_vel, pin_acc

    # Create a quaternion object from the Mujoco orientation (scalar-first)
    q = quaternion(*pin_pos[3:7])
    # Obtain the corresponding rotation matrix
    R = as_rotation_matrix(q)
    # Rotate the world frame linear velocity to the local frame
    pin_vel[0:3] = R.T @ pin_vel[0:3]

    # Reorder quaternion from scalar-first (Mujoco) to scalar-last (Pinocchio)
    pin_pos[[3, 4, 5, 6]] = pin_pos[[4, 5, 6, 3]]

    if qacc is not None:
        # Rotate the world frame linear acceleration to the local frame
        pin_acc[0:3] = R.T @ pin_acc[0:3]
        return pin_pos, pin_vel, pin_acc

    return pin_pos, pin_vel


def pin2muj(pin_pos: np.ndarray, pin_vel: np.ndarray, pin_acc: np.ndarray | None = None) -> tuple:
    """
    Converts Pinocchio state to Mujoco state by adjusting the quaternion representation and rotating the velocity.

    This function assumes that the quaternion representation of orientation in the Pinocchio state uses a scalar-last
    format (x, y, z, w), while the Mujoco state uses a scalar-first format (w, x, y, z). It also rotates the local
    frame linear velocity to the world frame.

    Args:
        pin_pos (numpy.ndarray): Pinocchio qpos array, which includes position and orientation.
        pin_vel (numpy.ndarray): Pinocchio qvel array, which includes linear and angular velocity.
        pin_acc (numpy.ndarray, optional): Pinocchio qacc array, which includes linear and angular acceleration.

    Returns:
        tuple: A tuple containing two numpy.ndarrays:
            - qpos (numpy.ndarray): Mujoco qpos array, with adjusted quaternion and position.
            - qvel (numpy.ndarray): Mujoco qvel array, with velocity rotated to the world frame.
    """
    # Copy the position and velocity to avoid modifying the original arrays
    qpos = pin_pos.copy()
    qvel = pin_vel.copy()
    qacc = pin_acc.copy() if pin_acc is not None else None

    if len(pin_pos) == len(qvel):
        if qacc is None:
            return qpos, qvel

        return qpos, qvel, qacc

    # Reorder quaternion from scalar-last (Pinocchio) to scalar-first (Mujoco)
    qpos[[3, 4, 5, 6]] = qpos[[6, 3, 4, 5]]

    # Create a quaternion object from the Pinocchio orientation (scalar-last)
    q = quaternion(*qpos[3:7])
    # Obtain the corresponding rotation matrix
    R = as_rotation_matrix(q)
    # Rotate the local frame linear velocity to the world frame
    qvel[0:3] = R @ qvel[0:3]

    if qacc is not None:
        qacc[0:3] = R @ qacc[0:3]
        return qpos, qvel, qacc

    return qpos, qvel


def mjx2mujoco(mj_model, mjx_model) -> mujoco.MjModel:
    """Update the mujoco model with the parameters from the mjx model.

    Args:
        mj_model (mujoco.MjModel): The mujoco model to be updated.
        mjx_model (mjx.Model): The mjx model containing the updated parameters

    Returns:
        mujoco.MjModel: The copy of the mujoco model with the updated parameters.
    """
    mj_model = copy.deepcopy(mj_model)

    # update dof_damping and dof_frictionloss
    mj_model.dof_damping = mjx_model.dof_damping
    mj_model.dof_frictionloss = mjx_model.dof_frictionloss

    # update bodies parameters
    mj_model.body_mass = np.array(mjx_model.body_mass)
    for i in range(mj_model.nbody):
        # update mass
        mj_model.body(i).mass = mjx_model.body_mass[i].item()
        # update inertia
        mj_model.body(i).inertia = mjx_model.body_inertia[i]
        # update quaternion
        mj_model.body(i).iquat = mjx_model.body_iquat[i]

    return mj_model
