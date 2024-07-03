import numpy as np
import mujoco
from mujoco_sysid import regressors
from mujoco_sysid.utils import muj2pin
from mujoco_sysid.parameters import get_dynamic_parameters

np.random.seed(0)


def test_body_regressor():
    import pinocchio as pin

    SAMPLES = 100

    for _ in range(SAMPLES):
        spatial_v, spatial_dv = np.random.rand(6), np.random.rand(6)

        pinY = pin.bodyRegressor(pin.Motion(spatial_v), pin.Motion(spatial_dv))

        mjY = regressors.body_regressor(spatial_v[:3], spatial_v[3:], spatial_dv[:3], spatial_dv[3:])

        assert np.allclose(mjY, pinY, atol=1e-6)


def test_joint_body_regressor():
    import pinocchio as pin

    # Test for fixed base manipulator
    from robot_descriptions.z1_description import URDF_PATH
    from robot_descriptions.z1_mj_description import MJCF_PATH

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    pindata = pinmodel.createData()

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjdata = mujoco.MjData(mjmodel)

    SAMPLES = 10000

    for _ in range(SAMPLES):
        q, v, dv = np.random.rand(pinmodel.nq), np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)
        pin.rnea(pinmodel, pindata, q, v, dv)

        mjdata.qpos[:] = q
        mjdata.qvel[:] = v
        mjdata.qacc[:] = dv
        mujoco.mj_inverse(mjmodel, mjdata)

        for jnt_id in range(pinmodel.njoints - 1):
            pinY = pin.jointBodyRegressor(pinmodel, pindata, jnt_id + 1)
            mjY = regressors.joint_body_regressor(mjmodel, mjdata, mjmodel.jnt_bodyid[jnt_id])

            assert np.allclose(mjY, pinY, atol=1e-6), f"Joint {jnt_id} failed. Pinocchio: {pinY}, Mujoco: {mjY}"

    # Test for the floating base joint
    from robot_descriptions.skydio_x2_description import URDF_PATH
    from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    pindata = pinmodel.createData()

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjdata = mujoco.MjData(mjmodel)

    SAMPLES = 10000

    for _ in range(SAMPLES):
        q = np.random.rand(pinmodel.nq)
        # normalize the quaternion
        q[3:7] /= np.linalg.norm(q[3:7])
        v, dv = np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)

        pinq, pinv, pindv = muj2pin(q, v, dv)

        mjdata.qpos[:] = q
        mjdata.qvel[:] = v
        mjdata.qacc[:] = dv
        mujoco.mj_inverse(mjmodel, mjdata)

        pin.rnea(pinmodel, pindata, pinq, pinv, pindv)

        for jnt_id in range(pinmodel.njoints - 1):
            pinY = pin.jointBodyRegressor(pinmodel, pindata, jnt_id + 1)

            mjY = regressors.joint_body_regressor(mjmodel, mjdata, mjmodel.jnt_bodyid[jnt_id])

            assert np.allclose(mjY, pinY, atol=1e-6), f"Joint {jnt_id} failed. Norm diff: {np.linalg.norm(mjY - pinY)}"


if __name__ == "__main__":
    test_body_regressor()
    test_joint_body_regressor()
