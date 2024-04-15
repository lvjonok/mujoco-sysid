"""This module contains functions with easier access to dynamics quantities"""

import mujoco
import numpy as np


def spacial_vel(model, data, body_idx: int, flg_local: int = 1):
    rotlin_vel = np.zeros(6)

    mujoco.mj_objectVelocity(model, data, 2, body_idx + 1, rotlin_vel, flg_local)

    # by default it is rotational, then linear
    linrot_vel = rotlin_vel.copy()
    linrot_vel[:3], linrot_vel[3:] = rotlin_vel[3:], rotlin_vel[:3]

    return linrot_vel


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def spacial_acc(model, data, body_idx: int, flg_local: int = 1):
    """Get spacial acceleration of chosen XBODY

    We assume that `mujoco.mj_rnePostConstraint(mj_model, mj_data)`
    is already called in prior to this function

    Args:
        model (_type_): _description_
        data (_type_): _description_
        body_idx (int): _description_
        flg_local (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    rotlin_acc = np.zeros(6)

    mujoco.mj_objectAcceleration(model, data, 2, body_idx + 1, rotlin_acc, flg_local)

    # by default it is rotational, then linear
    linrot_acc = rotlin_acc.copy()
    linrot_acc[:3], linrot_acc[3:] = rotlin_acc[3:], rotlin_acc[:3]

    rotation = data.xmat[body_idx + 1].reshape(3, 3).T.copy()

    vel = spacial_vel(model, data, body_idx, flg_local)

    # compensate
    linrot_acc[:3] += rotation @ model.opt.gravity - skew(vel[3:]) @ vel[:3]

    return linrot_acc


def Y_body(v_lin, v_ang, a_lin, a_ang):
    # Derivation is here: https://colab.research.google.com/drive/1xFte2FT0nQ0ePs02BoOx4CmLLw5U-OUZ?usp=sharing
    # this is identical to the pinnochio: pin.bodyRegressor(*[v_lin, v_ang], *[a_lin, a_ang])
    v1, v2, v3 = v_lin
    v4, v5, v6 = v_ang

    a1, a2, a3 = a_lin
    a4, a5, a6 = a_ang

    # fmt: off
    Y = np.array([
        [a1 - v2*v6 + v3*v5, -v5**2 - v6**2, -a6 + v4*v5, a5 + v4*v6, 0, 0, 0, 0, 0, 0],
        [a2 + v1*v6 - v3*v4, a6 + v4*v5, -v4**2 - v6**2, -a4 + v5*v6, 0, 0, 0, 0, 0, 0],
        [a3 - v1*v5 + v2*v4, -a5 + v4*v6, a4 + v5*v6, -v4**2 - v5**2, 0, 0, 0, 0, 0, 0],
        [0, 0, a3 - v1*v5 + v2*v4, -a2 - v1*v6 + v3*v4, a4, a5 - v4*v6, -v5*v6, a6 + v4*v5, v5**2 - v6**2, v5*v6],
        [0, -a3 + v1*v5 - v2*v4, 0, a1 - v2*v6 + v3*v5, v4*v6, a4 + v5*v6, a5, -v4**2 + v6**2, a6 - v4*v5, -v4*v6],
        [0, a2 + v1*v6 - v3*v4, -a1 + v2*v6 - v3*v5, 0, -v4*v5, v4**2 - v5**2, v4*v5, a4 - v5*v6, a5 + v4*v6, a6]
    ])
    # fmt: on

    return Y


def mj_bodyRegressor(mj_model, mj_data, body):
    # TODO:
    # Clarify more about transformations:
    # https://mujoco.readthedocs.io/en/stable/programming/simulation.html#coordinate-frames-and-transformations
    # https://royfeatherstone.org/teaching/index.html
    # https://royfeatherstone.org/teaching/2008/slides.pdf

    body_id = body + 1
    velocity = np.zeros(6)
    accel = np.zeros(6)
    _cross = np.zeros(3)

    mujoco.mj_objectVelocity(mj_model, mj_data, 2, body_id, velocity, 1)
    mujoco.mj_rnePostConstraint(mj_model, mj_data)
    mujoco.mj_objectAcceleration(mj_model, mj_data, 2, body_id, accel, 1)
    # rotation = mj_data.xmat[body_id].reshape(3, 3).copy()

    v, w = velocity[3:], velocity[:3]
    dv, dw = accel[3:], accel[:3]  # dv - classical acceleration, already containt g

    mujoco.mju_cross(_cross, w, v)

    dv -= _cross

    Y = Y_body(v, w, dv, dw)
    return Y


def jacobian(model, data, idx):
    jac_lin = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    rotation = data.xmat[idx + 1].reshape(3, 3).T.copy()

    mujoco.mj_jacBody(
        model,
        data,
        jac_lin,
        jac_rot,
        idx + 1,
    )

    return np.vstack((rotation @ jac_lin, rotation @ jac_rot))


def mj_jointRegressor(mj_model, mj_data):
    # what the hell is going on with indeces?
    # increase time in computation of multiplication between rotation
    # Investigate spatial tranfromation mju_transformSpatial t

    njoints = mj_model.njnt
    body_regressors = np.zeros((6 * njoints, njoints * 10))
    col_jac = np.zeros((6 * njoints, mj_model.nv))

    for i in range(njoints):
        # calculate cody regressors
        body_regressors[6 * i : 6 * (i + 1), 10 * i : 10 * (i + 1)] = mj_bodyRegressor(
            mj_model, mj_data, i + 1
        )

        col_jac[6 * i : 6 * i + 6, :] = jacobian(mj_model, mj_data, i + 1)

    joint_regressor = col_jac.T @ body_regressors
    return joint_regressor
