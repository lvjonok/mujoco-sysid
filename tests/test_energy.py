import mujoco
import numpy as np

from mujoco_sysid import parameters, regressors


def test_energy_match_z1():
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # enable energy
    mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    # disable friction, contact and limits
    mjmodel.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    mjdata = mujoco.MjData(mjmodel)

    # generate random samples of configuration and velocity
    N_SAMPLES = 1000
    for _ in range(N_SAMPLES):
        q, v = np.random.randn(mjmodel.nq), np.zeros(mjmodel.nv)

        mjdata.qpos[:] = q.copy()
        mjdata.qvel[:] = v.copy()

        mujoco.mj_step(mjmodel, mjdata)

        mj_en = mjdata.energy.copy()
        mj_en[0] += regressors.potential_energy_bias(mjmodel)

        # energy we expect and computed through simulator
        expected_energy = np.sum(mj_en)

        # get vector of dynamic parameters
        theta = np.concatenate([parameters.get_dynamic_parameters(mjmodel, i) for i in mjmodel.jnt_bodyid])
        # compute regressor of the total energy
        reg_en = regressors.mj_energyRegressor(mjmodel, mjdata)[2]

        computed_energy = reg_en @ theta

        assert np.isclose(
            expected_energy, computed_energy, atol=1e-6
        ), f"Expected energy: {expected_energy}, Computed energy: {computed_energy}"


def test_energy_match_skydio():
    from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # enable energy
    mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    # disable friction, contact and limits
    mjmodel.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    mjdata = mujoco.MjData(mjmodel)

    # generate random samples of configuration and velocity
    N_SAMPLES = 1000
    for _ in range(N_SAMPLES):
        q, v = np.random.randn(mjmodel.nq), np.zeros(mjmodel.nv)

        mjdata.qpos[:] = q.copy()
        mjdata.qvel[:] = v.copy()

        mujoco.mj_step(mjmodel, mjdata)

        mj_en = mjdata.energy.copy()
        mj_en[0] += regressors.potential_energy_bias(mjmodel)

        # energy we expect and computed through simulator
        expected_energy = np.sum(mj_en)

        # get vector of dynamic parameters
        theta = np.concatenate([parameters.get_dynamic_parameters(mjmodel, i) for i in mjmodel.jnt_bodyid])
        # compute regressor of the total energy
        reg_en = regressors.mj_energyRegressor(mjmodel, mjdata)

        computed_energy = reg_en @ theta

        assert np.isclose(
            expected_energy, computed_energy, atol=1e-6
        ), f"Expected energy: {expected_energy}, Computed energy: {computed_energy}"
