import mujoco
import numpy as np
import numpy.typing as npt


def skew(vector):
    return np.cross(np.eye(vector.size), vector.reshape(-1))


def get_dynamic_parameters(mjmodel, body_id) -> npt.ArrayLike:
    """Get the dynamic parameters \theta of a body
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body

    Returns:
        npt.ArrayLike: theta of the body
    """
    mass = mjmodel.body(body_id).mass
    rc = mjmodel.body(body_id).ipos
    diag_inertia = mjmodel.body(body_id).inertia

    # get the orientation of the body
    r_flat = np.zeros(9)
    mujoco.mju_quat2Mat(r_flat, mjmodel.body(body_id).iquat)

    R = r_flat.reshape(3, 3)

    shift = -mass * skew(rc) @ skew(rc)
    mjinertia = R.T @ np.diag(diag_inertia) @ R + shift

    return np.concatenate([mass, mass * rc, mjinertia[np.triu_indices(3)]])


def set_dynamic_parameters(mjmodel, body_id, theta: npt.ArrayLike) -> None:
    """Set the dynamic parameters to a body

    Args:
        mjmodel (mujoco.MjModel): The mujoco model
        body_id (int): The id of the body
        theta (npt.ArrayLike): The dynamic parameters of the body
    """

    mass = theta[0]
    rc = theta[1:4] / mass
    inertia = theta[4:]
    inertia_full = np.zeros((3, 3))
    inertia_full[np.triu_indices(3)] = inertia

    # shift the inertia
    inertia_full -= -mass * skew(rc) @ skew(rc)

    # eigen decomposition
    eigval, eigvec = np.linalg.eig(inertia_full)
    R = eigvec
    diag_inertia = eigval

    # check if singular, then abort
    if np.any(np.isclose(diag_inertia, 0)):
        raise ValueError("Cannot deduce inertia matrix because RIR^T is singular.")

    # set the mass
    mjmodel.body(body_id).mass = mass
    mjmodel.body(body_id).ipos = rc

    # set the orientation
    mujoco.mju_mat2Quat(mjmodel.body(body_id).iquat, R.flatten())

    # set the inertia
    mjmodel.body(body_id).inertia = diag_inertia
