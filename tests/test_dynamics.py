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
    # from robot_descriptions.z1_description import URDF_PATH
    # from robot_descriptions.z1_mj_description import MJCF_PATH

    # pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    # pindata = pinmodel.createData()

    # mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # mjdata = mujoco.MjData(mjmodel)

    # SAMPLES = 100

    # for _ in range(SAMPLES):
    #     q, v, dv = np.random.rand(pinmodel.nq), np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)
    #     pin.rnea(pinmodel, pindata, q, v, dv)

    #     mjdata.qpos[:] = q
    #     mjdata.qvel[:] = v
    #     mjdata.qacc[:] = dv
    #     mujoco.mj_inverse(mjmodel, mjdata)

    #     for jnt_id in range(pinmodel.njoints - 1):
    #         pinY = pin.jointBodyRegressor(pinmodel, pindata, jnt_id + 1)
    #         mjY = regressors.joint_body_regressor(mjmodel, mjdata, mjmodel.jnt_bodyid[jnt_id])

    #         assert np.allclose(mjY, pinY, atol=1e-6), f"Joint {jnt_id} failed. Pinocchio: {pinY}, Mujoco: {mjY}"

    # Test for the floating base joint
    from robot_descriptions.skydio_x2_description import URDF_PATH
    from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    pindata = pinmodel.createData()

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjdata = mujoco.MjData(mjmodel)

    SAMPLES = 1

    for _ in range(SAMPLES):
        # q = np.random.rand(pinmodel.nq)
        # normalize the quaternion
        # q[3:7] /= np.linalg.norm(q[3:7])
        q = np.array([0, 0, 1, 1, 0, 0, 0])
        v, dv = np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)

        v *= 0
        # dv *= 0
        pinq, pinv = muj2pin(q, v)

        # v = np.array([0, 0, 1, 0, 0, 0])
        # dv[3:6] *= 0

        print("Configuration")
        print(f"Pinocchio: {pinq}, {pinv}, {dv}")
        print(f"Mujoco: {q}, {v}, {dv}")

        pin.rnea(pinmodel, pindata, pinq, pinv, dv)

        mjdata.qpos[:] = q
        mjdata.qvel[:] = v
        mjdata.qacc[:] = dv
        mujoco.mj_inverse(mjmodel, mjdata)

        theta = get_dynamic_parameters(mjmodel, 1)

        print("Joint space forces")
        print(f"Pinocchio: {pindata.f[1]}")

        # rot = mjdata.ximat[1].reshape(3, 3).T
        print(f"Mujoco: {mjdata.qfrc_inverse}")

        print("Diff", pindata.f[1] - mjdata.qfrc_inverse)

        # {mjdata.qfrc_passive} {mjmodel.dof_armature} {mjdata.qfrc_constraint}")
        # print(rot)
        # print(np.concatenate((rot.T @ mjdata.qfrc_inverse[:3], rot.T @ mjdata.qfrc_inverse[3:])))

        for jnt_id in range(pinmodel.njoints - 1):
            pinY = pin.jointBodyRegressor(pinmodel, pindata, jnt_id + 1)

            # jac_lin = np.zeros((3, mjmodel.nv))
            # jac_rot = np.zeros((3, mjmodel.nv))
            # mujoco.mj_jacBody(mjmodel, mjdata, jac_lin, jac_rot, mjmodel.jnt_bodyid[jnt_id])

            # rotation = mjdata.xmat[mjmodel.jnt_bodyid[jnt_id]].reshape(3, 3)
            # jac = np.vstack((rotation.T @ jac_lin, rotation.T @ jac_rot))

            mjY = regressors.joint_body_regressor(mjmodel, mjdata, mjmodel.jnt_bodyid[jnt_id])
            # f = pindata.f[jnt_id + 1]
            # print(f"Joint {jnt_id} force: {f} mujoco force: {mjdata.qfrc_inverse}")

            assert np.allclose(
                pinY @ theta, np.array(pindata.f[jnt_id + 1]), atol=1e-6
            ), f"Joint {jnt_id} failed. Computed: {pinY @ theta}, Expected: {np.array(pindata.f[jnt_id + 1])}"

            print(f"Pinocchio force expected: {pinY @ theta} actual: {np.array(pindata.f[1])}")
            print(f"Mujoco force expected: {mjY @ theta} actual: {mjdata.qfrc_inverse}")

            assert np.allclose(
                mjY, pinY, atol=1e-6
            ), f"Joint {jnt_id} failed. Pinocchio: {pinY[:3, 0]}, Mujoco: {mjY[:3, 0]}"


if __name__ == "__main__":
    test_body_regressor()
    test_joint_body_regressor()
