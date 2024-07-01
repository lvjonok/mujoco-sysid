"""This module contains functions of the regressor representation of inverse dynamics for single rigid body:
    M_b * a_g + v x M_b * v = Y_b(v, a_g)*theta = f

and multiple bodies in generilized coordinates:
    M(q) * ddq + h(q, dq) = Y(q, dq, ddq)*theta = tau

with vector of inertial parameters (stacked for multiple bodies):
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

A linear parametrization of inverse dynamics is pivotal in SysID for robotic systems.
To the best of our knowledge, dedicated functions for this representation are not available in MuJoCo,
prompting us to develop this prototype.

References:
- Traversaro, Silvio, et al. "Identification of fully physical consistent inertial parameters using optimization on manifolds."
    2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016.
- Garofalo G, Ott C, Albu-Schäffer A. On the closed form computation of the dynamic matrices and their differentiations.
    In2013 IEEE/RSJ International Conference on Intelligent Robots and Systems 2013 Nov 3 (pp. 2364-2359). IEEE.
"""  # noqa

import mujoco
import numpy as np
from numpy import typing as npt


def body_regressor(
    v_lin: npt.ArrayLike, v_ang: npt.ArrayLike, a_lin: npt.ArrayLike, a_ang: npt.ArrayLike
) -> npt.ArrayLike:
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
    return np.array([
        [a1 - v2*v6 + v3*v5, -v5**2 - v6**2, -a6 + v4*v5, a5 + v4*v6, 0, 0, 0, 0, 0, 0],
        [a2 + v1*v6 - v3*v4, a6 + v4*v5, -v4**2 - v6**2, -a4 + v5*v6, 0, 0, 0, 0, 0, 0],
        [a3 - v1*v5 + v2*v4, -a5 + v4*v6, a4 + v5*v6, -v4**2 - v5**2, 0, 0, 0, 0, 0, 0],
        [0, 0, a3 - v1*v5 + v2*v4, -a2 - v1*v6 + v3*v4, a4, a5 - v4*v6, -v5*v6, a6 + v4*v5, v5**2 - v6**2, v5*v6],
        [0, -a3 + v1*v5 - v2*v4, 0, a1 - v2*v6 + v3*v5, v4*v6, a4 + v5*v6, a5, -v4**2 + v6**2, a6 - v4*v5, -v4*v6],
        [0, a2 + v1*v6 - v3*v4, -a1 + v2*v6 - v3*v5, 0, -v4*v5, v4**2 - v5**2, v4*v5, a4 - v5*v6, a5 + v4*v6, a6]
    ])
    # fmt: on


def joint_body_regressor(mj_model, mj_data, body_id) -> npt.ArrayLike:
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

    velocity = np.zeros(6)
    accel = np.zeros(6)
    _cross = np.zeros(3)

    mujoco.mj_objectVelocity(mj_model, mj_data, 2, body_id, velocity, 1)
    mujoco.mj_rnePostConstraint(mj_model, mj_data)
    mujoco.mj_objectAcceleration(mj_model, mj_data, 2, body_id, accel, 1)

    v, w = velocity[3:], velocity[:3]
    # dv - classical acceleration, already contains g
    dv, dw = accel[3:], accel[:3]
    mujoco.mju_cross(_cross, w, v)

    # if floating, should be cancelled
    # if mj_model.nq != mj_model.nv:

    rot = mj_data.xmat[body_id - 1].reshape(3, 3)
    # dv -= _cross
    # dv = rot @ dv

    return body_regressor(v, w, dv, dw)


def joint_torque_regressor(mj_model, mj_data) -> npt.ArrayLike:
    """mj_jointRegressor returns a regressor for the whole model

    This function calculates the regressor for the whole model in the MuJoCo model.

    This regressor is computed to use in joint-space calculations. It is a matrix that
    maps the inertial parameters of the bodies to the generalized forces.

    Newton-Euler equations for a rigid body are given by:
        M * a_g + v x M * v = f

    Expressing the spatial quantities in terms of the generalized quantities
    we can rewrite the equation for the system of bodies as:
        M * q_dot_dot + h = tau

    Where
        M is the mass matrix
        h is the bias term
        tau is the generalized forces

    Then, the regressor is a matrix Y such that:
        Y * theta = tau

    where:
        theta is the vector of inertial parameters of the bodies (10 parameters per body):
            theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]


    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        body_offset (int, optional): Starting index of the body, useful when some dummy bodies are introduced.

    Returns:
        npt.ArrayLike: regressor for the whole model
    """

    njoints = mj_model.njnt
    body_regressors = np.zeros((6 * njoints, njoints * 10))
    col_jac = np.zeros((6 * njoints, mj_model.nv))
    jac_lin = np.zeros((3, mj_model.nv))
    jac_rot = np.zeros((3, mj_model.nv))

    for i in range(njoints):
        # calculate cody regressors
        body_regressors[6 * i : 6 * (i + 1), 10 * i : 10 * (i + 1)] = joint_body_regressor(
            mj_model, mj_data, i + body_offset
        )

        mujoco.mj_jacBody(mj_model, mj_data, jac_lin, jac_rot, i + body_offset)

        # Calculate jacobians
        rotation = mj_data.xmat[i + body_offset].reshape(3, 3).copy()
        col_jac[6 * i : 6 * i + 3, :] = rotation.T @ jac_lin.copy()
        col_jac[6 * i + 3 : 6 * i + 6, :] = rotation.T @ jac_rot.copy()

    return col_jac.T @ body_regressors


def body_energyRegressor(
    v: npt.ArrayLike,
    w: npt.ArrayLike,
    r: npt.ArrayLike,
    R: npt.ArrayLike,
    gravity: npt.ArrayLike | None = None,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Computes the kinetic and potential energy regressors for a single body.

    This function calculates the energy regressors for the whole model in the MuJoCo model.

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
        gravity = np.array([0, 0, 9.81])

    kinetic = np.array(
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

    potential = np.array([*(gravity.T @ r).flatten(), *(gravity.T @ R).flatten(), 0, 0, 0, 0, 0, 0])

    return kinetic, potential


def get_jacobian(mjmodel, mjdata, bodyid):
    R = mjdata.xmat[bodyid].reshape(3, 3)

    jacp, jacr = np.zeros((3, 6)), np.zeros((3, 6))
    mujoco.mj_jacBody(mjmodel, mjdata, jacp, jacr, bodyid)

    return np.vstack([R.T @ jacr, R.T @ jacp])


def mj_energyRegressor(mj_model, mj_data) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    mj_energyRegressor returns kinetic, potential, and total energy regressors for the whole model.

    The energy regressors Y_k and Y_u are matrices such that:
        K = Y_k * theta
        U = Y_u * theta

    where:
        theta is the vector of inertial parameters of the bodies (10 parameters per body):
            theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

    The total energy regressor Y_e is simply the sum of Y_k and Y_u:
        E = K + U = Y_e * theta

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data
        body_offset (int, optional): Starting index of the body, useful when some dummy bodies are introduced. Defaults to 0.

    Returns:
        tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]: kinetic, potential, and total energy regressors for the whole model
    """  # noqa

    njoints = mj_model.njnt
    energy_regressor = np.zeros(njoints * 10)
    kinetic_regressor = np.zeros(njoints * 10)
    potential_regressor = np.zeros(njoints * 10)
    velocity = np.zeros(6)

    for i, bodyid in enumerate(mj_model.jnt_bodyid):
        # same as jacobian @ qvel
        mujoco.mj_objectVelocity(mj_model, mj_data, 2, bodyid, velocity, 1)
        v, w = velocity[3:], velocity[:3]

        rotation = mj_data.xmat[bodyid].reshape(3, 3)
        position = mj_data.xpos[bodyid]

        kinetic, potential = body_energyRegressor(v, w, position, rotation)
        kinetic_regressor[10 * i : 10 * (i + 1)] = kinetic
        potential_regressor[10 * i : 10 * (i + 1)] = potential
        energy_regressor[10 * i : 10 * (i + 1)] = kinetic + potential

    return kinetic_regressor, potential_regressor, energy_regressor


def potential_energy_bias(mjmodel):
    """
    The bodies before the first joint are considered to be fixed in space.
    They are included in potential energy calculation, but not in the regressor.
    """

    bias = 0
    for i in range(mjmodel.nbody):
        if i not in mjmodel.jnt_bodyid:
            bias += mjmodel.body(i).mass[0] * mjmodel.opt.gravity[2] * mjmodel.body(i).ipos[2]

    return bias
