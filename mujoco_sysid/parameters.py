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
