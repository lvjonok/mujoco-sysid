import mujoco
import numpy as np

from mujoco_sysid import parameters


def test_get_dynamic_parameters():
    # get z1 manipulator model
    import pinocchio as pin
    from robot_descriptions.z1_description import URDF_PATH
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    pinmodel = pin.buildModelFromUrdf(URDF_PATH)

    for i in range(6):
        pinparams = pinmodel.inertias[i].toDynamicParameters()
        mjparams = parameters.get_dynamic_parameters(mjmodel, i + 1)

        # check that mass is close
        assert np.isclose(pinparams[0], mjparams[0])

        # check that the center of mass is close
        assert np.allclose(pinparams[1:4], mjparams[1:4])

        # check that the inertia matrix is close
        assert np.allclose(pinparams[4:], mjparams[4:], atol=1e-6), "Inertia matrix is not the same"


def test_set_dynamic_parameters():
    import numpy as np
    from robot_descriptions.z1_mj_description import MJCF_PATH

    # Load the model
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    param = parameters.get_dynamic_parameters(model, 1)
    parameters.set_dynamic_parameters(model, 1, param)

    next_param = parameters.get_dynamic_parameters(model, 1)

    # mass is the same
    assert np.isclose(param[0], next_param[0])

    # center of mass is the same
    assert np.allclose(param[1:4], next_param[1:4])

    # inertia matrix is the same
    assert np.allclose(param[4:], next_param[4:], atol=1e-6), "Inertia matrix is not the same"
