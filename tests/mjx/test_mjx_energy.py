import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from mujoco_sysid.mjx import parameters, regressors

key = jax.random.PRNGKey(0)


def test_energy_match_z1():
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # alter the model so it becomes mjx compatible
    mjmodel.dof_frictionloss = 0
    mjmodel.opt.integrator = 0
    mjdata = mujoco.MjData(mjmodel)

    mjxmodel = mjx.put_model(mjmodel)
    print(mjxmodel.jnt_bodyid)
    assert len(mjxmodel.jnt_bodyid) == mjmodel.njnt, f"{len(mjxmodel.jnt_bodyid)} != {mjmodel.njnt}"
    mjxdata = mjx.put_data(mjmodel, mjdata)

    # enable energy
    mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    # disable friction, contact and limits
    mjmodel.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )

    # get vector of dynamic parameters
    theta = jnp.concatenate([parameters.get_dynamic_parameters(mjxmodel, i) for i in mjxmodel.jnt_bodyid])
    assert theta.shape == (mjxmodel.njnt * 10,)

    # generate random samples of configuration and velocity
    N_SAMPLES = 1000
    for _ in range(N_SAMPLES):
        q, v = jax.random.normal(key, (mjmodel.nq,)), jnp.zeros(mjmodel.nv)

        mjdata.qpos[:] = q.copy()
        mjdata.qvel[:] = v.copy()

        mujoco.mj_step(mjmodel, mjdata)

        mj_en = mjdata.energy.copy()
        mj_en[0] += regressors.potential_energy_bias(mjmodel)

        # energy we expect and computed through simulator
        expected_energy = np.sum(mj_en)

        # compute regressor of the total energy
        mjxdata = mjx.step(mjxmodel, mjxdata.replace(qpos=q, qvel=v))
        reg_en = regressors.energy_regressor(mjxmodel, mjxdata)[2]

        computed_energy = reg_en @ theta

        assert np.isclose(
            expected_energy, computed_energy, atol=1e-6
        ), f"Expected energy: {expected_energy}, Computed energy: {computed_energy}"


# def test_energy_match_skydio():
#     from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

#     mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
#     # enable energy
#     mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
#     # disable friction, contact and limits
#     mjmodel.opt.disableflags |= (
#         mujoco.mjtDisableBit.mjDSBL_CONTACT
#         | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
#         | mujoco.mjtDisableBit.mjDSBL_LIMIT
#     )
#     mjdata = mujoco.MjData(mjmodel)

#     # generate random samples of configuration and velocity
#     N_SAMPLES = 1000
#     for _ in range(N_SAMPLES):
#         q, v = np.random.randn(mjmodel.nq), np.zeros(mjmodel.nv)

#         mjdata.qpos[:] = q.copy()
#         mjdata.qvel[:] = v.copy()

#         mujoco.mj_step(mjmodel, mjdata)

#         mj_en = mjdata.energy.copy()
#         mj_en[0] += regressors.potential_energy_bias(mjmodel)

#         # energy we expect and computed through simulator
#         expected_energy = np.sum(mj_en)

#         # get vector of dynamic parameters
#         theta = np.concatenate([parameters.get_dynamic_parameters(mjmodel, i) for i in mjmodel.jnt_bodyid])
#         # compute regressor of the total energy
#         reg_en = regressors.mj_energyRegressor(mjmodel, mjdata)[2]

#         computed_energy = reg_en @ theta

#         assert np.isclose(
#             expected_energy, computed_energy, atol=1e-6
#         ), f"Expected energy: {expected_energy}, Computed energy: {computed_energy}"
