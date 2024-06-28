import numpy as np
from quaternion import as_rotation_matrix, quaternion


def muj2pin(qpos: np.ndarray, qvel: np.ndarray) -> tuple:
    """
    Converts Mujoco state to Pinocchio state by adjusting the quaternion representation and rotating the velocity.

    This function assumes that the quaternion representation of orientation in the Mujoco state uses a scalar-first
    format (w, x, y, z), while the Pinocchio state uses a scalar-last format (x, y, z, w). It also rotates the linear
    velocity from the world frame to the local frame.

    Args:
        qpos (numpy.ndarray): Mujoco qpos array, which includes position and orientation.
        qvel (numpy.ndarray): Mujoco qvel array, which includes linear and angular velocity.

    Returns:
        tuple: A tuple containing two numpy.ndarrays:
            - pin_pos (numpy.ndarray): Pinocchio qpos array, with adjusted quaternion and position.
            - pin_vel (numpy.ndarray): Pinocchio qvel array, with velocity rotated to the local frame.
    """
    # Copy the position and velocity to avoid modifying the original arrays
    pin_pos = qpos.copy()
    pin_vel = qvel.copy()

    if len(pin_pos) == len(qvel):
        return pin_pos, pin_vel

    # Create a quaternion object from the Mujoco orientation (scalar-first)
    q = quaternion(*pin_pos[3:7])
    # Obtain the corresponding rotation matrix
    R = as_rotation_matrix(q)
    # Rotate the world frame linear velocity to the local frame
    pin_vel[0:3] = R.T @ pin_vel[0:3]

    # Reorder quaternion from scalar-first (Mujoco) to scalar-last (Pinocchio)
    pin_pos[[3, 4, 5, 6]] = pin_pos[[4, 5, 6, 3]]

    return pin_pos, pin_vel


def pin2muj(pin_pos: np.ndarray, pin_vel: np.ndarray) -> tuple:
    """
    Converts Pinocchio state to Mujoco state by adjusting the quaternion representation and rotating the velocity.

    This function assumes that the quaternion representation of orientation in the Pinocchio state uses a scalar-last
    format (x, y, z, w), while the Mujoco state uses a scalar-first format (w, x, y, z). It also rotates the local
    frame linear velocity to the world frame.

    Args:
        pin_pos (numpy.ndarray): Pinocchio qpos array, which includes position and orientation.
        pin_vel (numpy.ndarray): Pinocchio qvel array, which includes linear and angular velocity.

    Returns:
        tuple: A tuple containing two numpy.ndarrays:
            - qpos (numpy.ndarray): Mujoco qpos array, with adjusted quaternion and position.
            - qvel (numpy.ndarray): Mujoco qvel array, with velocity rotated to the world frame.
    """
    # Copy the position and velocity to avoid modifying the original arrays
    qpos = pin_pos.copy()
    qvel = pin_vel.copy()

    if len(pin_pos) == len(qvel):
        return qpos, qvel

    # Reorder quaternion from scalar-last (Pinocchio) to scalar-first (Mujoco)
    qpos[[3, 4, 5, 6]] = qpos[[6, 3, 4, 5]]

    # Create a quaternion object from the Pinocchio orientation (scalar-last)
    q = quaternion(*qpos[3:7])
    # Obtain the corresponding rotation matrix
    R = as_rotation_matrix(q)
    # Rotate the local frame linear velocity to the world frame
    qvel[0:3] = R @ qvel[0:3]

    return qpos, qvel
