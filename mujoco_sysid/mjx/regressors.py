import jax
import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx
from mujoco.mjx._src import scan
from mujoco.mjx._src.math import transform_motion
from mujoco.mjx._src.types import DisableBit


def object_velocity(mjx_model: mjx.Model, mjx_data: mjx.Data, bodyid) -> jpt.ArrayLike:
    pos = mjx_data.xpos[bodyid]
    rot = mjx_data.xmat[bodyid]  # TODO: maybe reshape is required here

    # transform spatial
    vel = mjx_data.cvel[bodyid]
    oldpos = mjx_data.subtree_com[mjx_model.body_rootid[bodyid]]

    return transform_motion(vel, pos - oldpos, rot)


def mjx_rnePostConstraint(m: mjx.Model, d: mjx.Data) -> jpt.ArrayLike:
    nbody = m.nbody
    cfrc_com = jnp.zeros(6)
    cfrc = jnp.zeros(6)
    lfrc = jnp.zeros(6)

    all_cacc = jnp.zeros((nbody, 6))

    # clear cacc, set world acceleration to -gravity
    if not m.opt.disableflags & DisableBit.GRAVITY:
        cacc = jnp.concatenate((jnp.zeros((nbody, 3)), -m.opt.gravity), axis=1)

    # FIXME: assumption that xfrc_applied is zero
    # FIXME: assumption that contacts are zero
    # FIXME: assumption that connect and weld constraints are zero

    # forward pass over bodies: compute acc
    cacc = jnp.zeros(6)
    for j in range(nbody):
        bda = m.body_dofadr[j]

        # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
        cacc = all_cacc[m.body_parentid[j]] + d.cdof_dot[bda] * d.qvel[bda] + d.cdof[bda] * d.qacc[bda]


def com_acc(m: mjx.Model, d: mjx.Data) -> jpt.ArrayLike:
    # forward scan over tree: accumulate link center of mass acceleration
    def cacc_fn(cacc, cdof_dot, qvel):
        if cacc is None:
            if m.opt.disableflags & DisableBit.GRAVITY:
                cacc = jnp.zeros((6,))
            else:
                cacc = jnp.concatenate((jnp.zeros((3,)), -m.opt.gravity))

        cacc += jnp.sum(jax.vmap(jnp.multiply)(cdof_dot, qvel), axis=0)

        return cacc

    return scan.body_tree(m, cacc_fn, "vv", "b", d.cdof_dot, d.qvel)


def object_acceleration(mjx_model: mjx.Model, mjx_data: mjx.Data, bodyid) -> jpt.ArrayLike:
    pos = mjx_data.xpos[bodyid]
    rot = mjx_data.xmat[bodyid]  # TODO: maybe reshape is required here

    # transform spatial
    accel = mjx_data.cacc[bodyid]
    mjx_data.cvel
    oldpos = mjx_data.subtree_com[mjx_model.body_rootid[bodyid]]

    velocity = object_velocity(mjx_model, mjx_data, bodyid)

    # transform com-based acceleration to local frame
    acceleration = transform_motion(accel, pos - oldpos, rot)

    # acc_tran += vel_rot x vel_tran
    correction = jnp.cross(velocity[:3], velocity[3:])
    acceleration[3:] += correction

    return acceleration


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


def body_regressor(
    v_lin: jpt.ArrayLike,
    v_ang: jpt.ArrayLike,
    a_lin: jpt.ArrayLike,
    a_ang: jpt.ArrayLike,
) -> jpt.ArrayLike:
    """Y_body returns a regressor for a single rigid body

    Newton-Euler equations for a rigid body are given by:
    M * a_g + v x M * v = f

    where:
        M is the spatial inertia matrix of the body
        a_g is the acceleration of the body
        v is the spatial velocity of the body
        f is the spatial force acting on the body

    The regressor is a matrix Y such that:
        Y \theta = f

    where:
        \theta is the vector of inertial parameters of the body (10 parameters)

    More expressive derivation is given here:
        https://colab.research.google.com/drive/1xFte2FT0nQ0ePs02BoOx4CmLLw5U-OUZ?usp=sharing

    Args:
        v_lin (npt.ArrayLike): linear velocity of the body
        v_ang (npt.ArrayLike): angular velocity of the body
        a_lin (npt.ArrayLike): linear acceleration of the body
        a_ang (npt.ArrayLike): angular acceleration of the body

    Returns:
        npt.ArrayLike: regressor for the body
    """
    v1, v2, v3 = v_lin
    v4, v5, v6 = v_ang

    a1, a2, a3 = a_lin
    a4, a5, a6 = a_ang

    # fmt: off
    return jnp.array([
        [a1 - v2*v6 + v3*v5, -v5**2 - v6**2, -a6 + v4*v5, a5 + v4*v6, 0, 0, 0, 0, 0, 0],
        [a2 + v1*v6 - v3*v4, a6 + v4*v5, -v4**2 - v6**2, -a4 + v5*v6, 0, 0, 0, 0, 0, 0],
        [a3 - v1*v5 + v2*v4, -a5 + v4*v6, a4 + v5*v6, -v4**2 - v5**2, 0, 0, 0, 0, 0, 0],
        [0, 0, a3 - v1*v5 + v2*v4, -a2 - v1*v6 + v3*v4, a4, a5 - v4*v6, -v5*v6, a6 + v4*v5, v5**2 - v6**2, v5*v6],
        [0, -a3 + v1*v5 - v2*v4, 0, a1 - v2*v6 + v3*v5, v4*v6, a4 + v5*v6, a5, -v4**2 + v6**2, a6 - v4*v5, -v4*v6],
        [0, a2 + v1*v6 - v3*v4, -a1 + v2*v6 - v3*v5, 0, -v4*v5, v4**2 - v5**2, v4*v5, a4 - v5*v6, a5 + v4*v6, a6]
    ])
    # fmt: on


def joint_body_regressor(mjxmodel: mjx.Model, mjxdata: mjx.Data, bodyid) -> jpt.ArrayLike:
    """mj_bodyRegressor returns a regressor for a single rigid body

    This function calculates the regressor for a single rigid body in the MuJoCo model.
    Given the index of body we compute the velocity and acceleration of the body and
    then calculate the regressor using the Y_body function.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        body_id: ID of the body

    Returns:
        npt.ArrayLike: regressor for the body
    """
    velocity = object_velocity(mjxmodel, mjxdata, bodyid)
    # accel = mjx.

    # velocity = np.zeros(6)
    # accel = np.zeros(6)
    # _cross = np.zeros(3)

    # mujoco.mj_objectVelocity(mj_model, mj_data, 2, body_id, velocity, 1)
    # mujoco.mj_objectAcceleration(mj_model, mj_data, 2, body_id, accel, 1)

    # v, w = velocity[3:], velocity[:3]
    # # dv - classical acceleration, already contains g
    # dv, dw = accel[3:], accel[:3]
    # mujoco.mju_cross(_cross, w, v)

    # # for floating base, this is already included in dv
    # if mj_model.nq == mj_model.nv:
    #     dv -= _cross

    # return body_regressor(v, w, dv, dw)
