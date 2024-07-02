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

    print('z1 matched succsessfully')

    # Test for the floating base joint
    from robot_descriptions.skydio_x2_description import URDF_PATH
    from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)
    pindata = pinmodel.createData()

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    for act_id in range(4):
        mjmodel.actuator(act_id).ctrlrange = np.array([-1e4, 1e4])
    mjdata = mujoco.MjData(mjmodel)
    theta = get_dynamic_parameters(mjmodel, 1)

    SAMPLES = 1

    for _ in range(SAMPLES):
        q = np.random.rand(pinmodel.nq)
        # normalize the quaternion
        q[3:7] /= np.linalg.norm(q[3:7])
        # q = np.array([0, 0, 1, 1, 0, 0, 0])
        v, dv = np.random.rand(pinmodel.nv), np.random.rand(pinmodel.nv)

        # v[:3] *= 0  # FIXME: occasionally when we set the linear velocity to zero, the test works
        # otherwise the inverse dynamics does not match

        pinq, pinv = muj2pin(q, v)

        print("Configuration")
        print(f"Pinocchio: {pinq}, {pinv}, {dv}")
        print(f"Mujoco: {q}, {v}, {dv}")

        # Selector matrix for actuators
        selector = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-0.18, 0.18, 0.18, -0.18],
                [0.14, 0.14, -0.14, -0.14],
                [-0.0201, 0.0201, 0.0201, -0.0201],
            ]
        )
        ctrl = np.random.randn(4)
        tau = selector @ ctrl
        dv = pin.aba(pinmodel, pindata, pinq, pinv, tau)

        result = pin.rnea(pinmodel, pindata, pinq, pinv, dv)

        mjdata.qpos[:] = q
        mjdata.qvel[:] = v
        # mjdata.qacc[:] = dv
        mjdata.ctrl[:] = ctrl

        mujoco.mj_step(mjmodel, mjdata)
        # mujoco.mj_inverse(mjmodel, mjdata)

        print("Joint space forces")
        print(f"Pinocchio: {pindata.f[1]} {result}")

        rot = mjdata.xmat[1].reshape(3, 3)
        print("rotation", rot)
        mujtau = mjdata.qfrc_actuator.copy()

        mujtau[:3] = rot.T @ mujtau[:3]
        print(f"Mujoco: {mjdata.qfrc_actuator} {mujtau} {mjdata.qfrc_inverse}")
        # jac_lin = np.zeros((3, mjmodel.nv))
        # jac_rot = np.zeros((3, mjmodel.nv))
        # mujoco.mj_jacBody(mjmodel, mjdata, jac_lin, jac_rot, mjmodel.jnt_bodyid[0])

        # rotation = mjdata.xmat[mjmodel.jnt_bodyid[0]].reshape(3, 3).T
        # print(f"rotation: {rotation}")
        # jac = np.vstack((rotation.T @ jac_lin, rotation.T @ jac_rot))

        # print("Test", jac @ mjdata.qfrc_inverse)
        # print(jac)

        # print("Diff", pindata.f[1] - mjdata.qfrc_inverse)
        # print(f"{mjdata.qfrc_passive} {mjmodel.dof_armature} {mjdata.qfrc_constraint}")
        # return

        # print(rot)
        # print(np.concatenate((rot.T @ mjdata.qfrc_inverse[:3], rot.T @ mjdata.qfrc_inverse[3:])))

        for jnt_id in range(pinmodel.njoints - 1):
            pinY = pin.jointBodyRegressor(pinmodel, pindata, jnt_id + 1)

            mjY = regressors.joint_body_regressor(mjmodel, mjdata, mjmodel.jnt_bodyid[jnt_id])
            # f = pindata.f[jnt_id + 1]
            # print(f"Joint {jnt_id} force: {f} mujoco force: {mjdata.qfrc_inverse}")

            assert np.allclose(
                pinY @ theta, np.array(pindata.f[jnt_id + 1]), atol=1e-6
            ), f"Joint {jnt_id} failed. Computed: {pinY @ theta}, Expected: {np.array(pindata.f[jnt_id + 1])}"

            print(f"Pinocchio force expected: {pinY @ theta} actual: {np.array(pindata.f[1])}")
            print(f"Mujoco force expected: {mjY @ theta} actual: {mujtau}")

            assert np.allclose(
                mjY, pinY, atol=1e-6
            ), f"Joint {jnt_id} failed. Pinocchio: {pinY[:3, 0]}, Mujoco: {mjY[:3, 0]}"


if __name__ == "__main__":
    test_body_regressor()
    test_joint_body_regressor()
