import jax.numpy as jnp
import mujoco
from mujoco import mjx

from mujoco_sysid.mjx import parameters


def test_mjx_get_dynamic_parameters():
    # get z1 manipulator model
    import pinocchio as pin
    from robot_descriptions.z1_description import URDF_PATH
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # alter the model so it becomes mjx compatible
    mjmodel.dof_frictionloss = 0
    mjmodel.opt.integrator = 1

    mjxmodel = mjx.put_model(mjmodel)

    pinmodel = pin.buildModelFromUrdf(URDF_PATH)

    for i in range(6):
        pinparams = pinmodel.inertias[i].toDynamicParameters()
        mjparams = parameters.get_dynamic_parameters(mjxmodel, i + 1)

        # check that mass is close
        assert jnp.isclose(pinparams[0], mjparams[0])

        # check that the center of mass is close
        assert jnp.allclose(pinparams[1:4], mjparams[1:4])

        # check that the inertia matrix is close
        assert jnp.allclose(pinparams[4:], mjparams[4:], atol=1e-6), "Inertia matrix is not the same"


def test_mjx_set_dynamic_parameters():
    import numpy as np
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # alter the model so it becomes mjx compatible
    mjmodel.dof_frictionloss = 0
    mjmodel.opt.integrator = 1

    mjxmodel = mjx.put_model(mjmodel)

    param = parameters.get_dynamic_parameters(mjxmodel, 1)
    parameters.set_dynamic_parameters(mjxmodel, 1, param)

    next_param = parameters.get_dynamic_parameters(mjxmodel, 1)

    # mass is the same
    assert np.isclose(param[0], next_param[0])

    # center of mass is the same
    assert np.allclose(param[1:4], next_param[1:4])

    # inertia matrix is the same
    assert np.allclose(param[4:], next_param[4:], atol=1e-6), "Inertia matrix is not the same"
