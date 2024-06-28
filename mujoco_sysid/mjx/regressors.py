import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx


def object_velocity(mjx_model: mjx.Model, mjx_data: mjx.Data, bodyid) -> jpt.ArrayLike:
    pos = mjx_data.xpos[bodyid]
    rot = mjx_data.xmat[bodyid]  # TODO: maybe reshape is required here

    # transform spatial
    vec = mjx_data.cvel[bodyid]
    newpos = pos
    oldpos = mjx_data.subtree_com[mjx_model.body_rootid[bodyid]]
    rotnew2old = rot

    # apply translation
    dif = newpos - oldpos
    cros = jnp.cross(dif, vec[3:])
    vec = vec.at[3:].set(vec[3:] + cros)

    # apply rotation
    res = jnp.zeros(6)
    res = res.at[:3].set(rotnew2old.T @ vec[:3])
    res = res.at[3:].set(rotnew2old.T @ vec[3:])

    return res


def energy_regressor(mjx_model: mjx.Model, mjx_data: mjx.Data) -> tuple[jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike]:
    """Get the energy regressor matrices

    The energy regressors Y_k and Y_u are matrices such that:
        K = Y_k * theta
        U = Y_u * theta

    where:
        theta is the vector of inertial parameters of the bodies (10 parameters per body):
            theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    The total energy regressor Y_e is simply the sum of Y_k and Y_u:
        E = K + U = Y_e * theta


    Args:
        mjx_model (mujoco.mjx.MjModel): The mujoco model
        mjx_data (mujoco.mjx.MjData): The mujoco data

    Returns:
        tuple[jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike]: The regressor matrices
    """

    njoints = mjx_model.njnt
    energy_regressor = jnp.zeros((njoints, 10 * mjx_model.nbody), dtype=jnp.float64)
    kinetic_regressor = jnp.zeros((njoints, 10 * mjx_model.nbody), dtype=jnp.float64)
    potential_regressor = jnp.zeros((njoints, 10 * mjx_model.nbody), dtype=jnp.float64)

    for i, bodyid in enumerate(mjx_model.jnt_bodyid):
        # get the spatial velocity
        pass
