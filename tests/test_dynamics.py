import numpy as np
import mujoco
from mujoco_sysid import regressors

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
    from robot_descriptions.z1_description import URDF_PATH
    from robot_descriptions.z1_mj_description import MJCF_PATH

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    pindata = pinmodel.createData()

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjdata = mujoco.MjData(mjmodel)

    SAMPLES = 100

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


if __name__ == "__main__":
    test_body_regressor()
    test_joint_body_regressor()
