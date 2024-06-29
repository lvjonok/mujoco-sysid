import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx
from mujoco.mjx._src.math import transform_motion


def object_velocity(mjx_model: mjx.Model, mjx_data: mjx.Data, bodyid) -> jpt.ArrayLike:
    pos = mjx_data.xpos[bodyid]
    rot = mjx_data.xmat[bodyid]  # TODO: maybe reshape is required here

    # transform spatial
    vel = mjx_data.cvel[bodyid]
    oldpos = mjx_data.subtree_com[mjx_model.body_rootid[bodyid]]

    return transform_motion(vel, pos - oldpos, rot)


def body_energyRegressor(
    v: jpt.ArrayLike,
    w: jpt.ArrayLike,
    r: jpt.ArrayLike,
    R: jpt.ArrayLike,
    gravity: jpt.ArrayLike | None = None,
) -> tuple[jpt.ArrayLike, jpt.ArrayLike]:
    """
    Computes the kinetic and potential energy regressors for a single body.

    The energy regressors are computed to represent the kinetic and potential energy of the system
    as a linear combination of the inertial parameters of the bodies.

    The kinetic energy of a rigid body is given by:
        K = 0.5 * v^T * M * v

    The potential energy of a rigid body in a uniform gravitational field is given by:
        U = m * g^T * r

    where:
        v is the spatial velocity of the body
        M is the spatial inertia matrix of the body
        m is the mass of the body
        g is the gravitational acceleration vector
        r is the position of the body's center of mass

    The energy regressors Y_k and Y_u are matrices such that:
        K = Y_k * theta
        U = Y_u * theta

    Args:
        v (npt.ArrayLike): Linear velocity of the body.
        w (npt.ArrayLike): Angular velocity of the body.
        r (npt.ArrayLike): Position of the body.
        R (npt.ArrayLike): Rotation matrix of the body.
        gravity (npt.ArrayLike, optional): Gravity vector. Defaults to np.array([0, 0, 9.81]).

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike]: Kinetic and potential energy regressors.
    """
    if gravity is None:
        gravity = jnp.array([0, 0, 9.81])

    kinetic = jnp.array(
        [
            0.5 * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2),
            -w[1] * v[2] + w[2] * v[1],
            w[0] * v[2] - w[2] * v[0],
            -w[0] * v[1] + w[1] * v[0],
            0.5 * w[0] ** 2,
            w[0] * w[1],
            0.5 * w[1] ** 2,
            w[0] * w[2],
            w[1] * w[2],
            0.5 * w[2] ** 2,
        ]
    )

    potential = jnp.array([*(gravity.T @ r).flatten(), *(gravity.T @ R).flatten(), 0, 0, 0, 0, 0, 0])

    return kinetic, potential


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
    # energy_regressor = jnp.zeros(njoints * 10)
    kinetic_regressor = jnp.zeros(njoints * 10)
    potential_regressor = jnp.zeros(njoints * 10)

    for jnt_id, bodyid in enumerate(mjx_model.jnt_bodyid):
        # get the spatial velocity
        velocity = object_velocity(mjx_model, mjx_data, bodyid)
        v, w = velocity[3:], velocity[:3]

        rot = mjx_data.xmat[bodyid]
        pos = mjx_data.xpos[bodyid]

        kinetic, potential = body_energyRegressor(v, w, pos, rot)
        kinetic_regressor = kinetic_regressor.at[10 * jnt_id : 10 * jnt_id + 10].set(kinetic)
        potential_regressor = potential_regressor.at[10 * jnt_id : 10 * jnt_id + 10].set(potential)

        # energy_regressor.at[10 * jnt_id : 10 * jnt_id + 10].set(kinetic + potential)

    energy_regressor = kinetic_regressor + potential_regressor

    return kinetic_regressor, potential_regressor, energy_regressor


def potential_energy_bias(mjxmodel: mjx.Model):
    """
    The bodies before the first joint are considered to be fixed in space.
    They are included in potential energy calculation, but not in the regressor.
    """

    bias = 0
    for i in range(mjxmodel.nbody):
        if i not in mjxmodel.jnt_bodyid:
            bias += mjxmodel.body_mass[i] * mjxmodel.opt.gravity[2] * mjxmodel.body_ipos[i][2]

    return bias
