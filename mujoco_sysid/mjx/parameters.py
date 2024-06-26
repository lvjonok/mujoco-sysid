import jax.numpy as jnp
import jax.typing as jpt
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx
from mujoco.mjx._src import math


def skew(vector: jpt.ArrayLike) -> jpt.ArrayLike:
    return jnp.cross(jnp.eye(vector.size), vector.reshape(-1))


def get_dynamic_parameters(mjx_model: mjx.Model, bodyid) -> jpt.ArrayLike:
    """Get the dynamic parameters \theta of a body
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    Args:
        mjx_model (mujoco.mjx.MjModel): The mujoco model
        bodyid (int): The id of the body

    Returns:
        jpt.ArrayLike: theta of the body
    """
    mass = mjx_model.body_mass[bodyid]
    rc = mjx_model.body_ipos[bodyid]
    diag_inertia = mjx_model.body_inertia[bodyid]

    # get the orientation of the body
    R = math.quat_to_mat(mjx_model.body_iquat[bodyid])

    shift = mass * skew(rc) @ skew(rc)
    mjinertia = R @ jnp.diag(diag_inertia) @ R.T - shift

    upper_triangular = jnp.array(
        [
            mjinertia[0, 0],
            mjinertia[0, 1],
            mjinertia[1, 1],
            mjinertia[0, 2],
            mjinertia[1, 2],
            mjinertia[2, 2],
        ]
    )

    return jnp.concatenate([jnp.array([mass]), mass * rc, upper_triangular])


def set_dynamic_parameters(mjx_model: mjx.Model, bodyid, theta: jpt.ArrayLike) -> mjx.Model:
    """Set the dynamic parameters to a body

    Args:
        mjx_model (mujoco.mjx.MjModel): The mujoco model
        bodyid (int): The id of the body
        theta (jpt.ArrayLike): The dynamic parameters of the body

    Returns:
        mjx.Model: The updated model
    """
    mass = theta[0]
    rc = theta[1:4] / mass
    inertia = theta[4:]
    inertia_full = jnp.array(
        [
            [inertia[0], inertia[1], inertia[3]],
            [inertia[1], inertia[2], inertia[4]],
            [inertia[3], inertia[4], inertia[5]],
        ]
    )

    # shift the inertia matrix
    inertia_full += mass * skew(rc) @ skew(rc)

    # eigen decomposition
    diag_inertia, R = jnp.linalg.eigh(inertia_full)

    # check if any of the eigenvalues are negative
    if jnp.any(diag_inertia < 0):
        raise ValueError("Inertia matrix is not positive definite")

    # convert the rotation matrix to quaternion
    quat = Rotation.from_matrix(R).as_quat()

    # update the body parameters
    return mjx_model.tree_replace(
        {
            "body_inertia": mjx_model.body_inertia.at[bodyid].set(diag_inertia),
            "body_iquat": mjx_model.body_iquat.at[bodyid].set(quat),
            "body_mass": mjx_model.body_mass.at[bodyid].set(mass),
        }
    )
