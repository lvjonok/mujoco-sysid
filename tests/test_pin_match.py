import pinocchio as pin
import mujoco
from robot_descriptions.z1_description import URDF_PATH
from robot_descriptions.z1_mj_description import MJCF_PATH
import numpy as np

from mujoco_sysid import spacial_vel, spacial_acc, jacobian, mj_jointRegressor

pinmodel = pin.buildModelFromUrdf(URDF_PATH)
pindata = pinmodel.createData()

mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
mjdata = mujoco.MjData(mjmodel)


def test_velocity():
    q = np.random.randn(mjmodel.nq)
    v = np.random.randn(mjmodel.nv)
    dv = np.random.randn(mjmodel.nv)

    mjdata.qpos[:] = q.copy()
    mjdata.qvel[:] = v.copy()
    mjdata.qacc[:] = dv.copy()

    pin.computeAllTerms(pinmodel, pindata, q, v)
    pin.forwardKinematics(pinmodel, pindata, q, v, dv)

    mujoco.mj_inverse(mjmodel, mjdata)

    for idx in range(mjmodel.nv):
        pinvel = pin.getVelocity(pinmodel, pindata, idx, pin.LOCAL)
        mjvel = spacial_vel(mjmodel, mjdata, idx)

        np.testing.assert_allclose(pinvel, mjvel, atol=1e-6)


def test_acceleration():
    q = np.random.randn(mjmodel.nq)
    v = np.random.randn(mjmodel.nv)
    dv = np.random.randn(mjmodel.nv)

    mjdata.qpos[:] = q.copy()
    mjdata.qvel[:] = v.copy()
    mjdata.qacc[:] = dv.copy()

    pin.computeAllTerms(pinmodel, pindata, q, v)
    pin.forwardKinematics(pinmodel, pindata, q, v, dv)

    mujoco.mj_inverse(mjmodel, mjdata)
    mujoco.mj_rnePostConstraint(mjmodel, mjdata)

    for idx in range(mjmodel.nv):
        pinacc = pin.getAcceleration(pinmodel, pindata, idx, pin.LOCAL)
        mjacc = spacial_acc(mjmodel, mjdata, idx)

        np.testing.assert_allclose(pinacc, mjacc, atol=1e-6)


def test_jacobian():
    q = np.random.randn(mjmodel.nq)
    v = np.random.randn(mjmodel.nv)
    dv = np.random.randn(mjmodel.nv)

    mjdata.qpos[:] = q.copy()
    mjdata.qvel[:] = v.copy()
    mjdata.qacc[:] = dv.copy()

    pin.computeAllTerms(pinmodel, pindata, q, v)
    pin.forwardKinematics(pinmodel, pindata, q, v, dv)

    mujoco.mj_inverse(mjmodel, mjdata)
    mujoco.mj_rnePostConstraint(mjmodel, mjdata)

    for idx in range(mjmodel.nv):
        jacobian_pin = pin.getJointJacobian(pinmodel, pindata, idx, pin.LOCAL)

        jacobian_mj = jacobian(mjmodel, mjdata, idx)

        np.testing.assert_allclose(jacobian_pin, jacobian_mj, atol=1e-6)


def test_joint_regressor():
    q = np.random.randn(mjmodel.nq)
    v = np.random.randn(mjmodel.nv)
    dv = np.random.randn(mjmodel.nv)

    mjdata.qpos[:] = q.copy()
    mjdata.qvel[:] = v.copy()
    mjdata.qacc[:] = dv.copy()

    pin.computeAllTerms(pinmodel, pindata, q, v)
    pin.forwardKinematics(pinmodel, pindata, q, v, dv)

    mujoco.mj_inverse(mjmodel, mjdata)
    mujoco.mj_rnePostConstraint(mjmodel, mjdata)

    pin_regressor = pin.computeJointTorqueRegressor(pinmodel, pindata, q, v, dv)
    mj_regressor = mj_jointRegressor(mjmodel, mjdata)

    np.testing.assert_allclose(pin_regressor, mj_regressor, atol=1e-6)
