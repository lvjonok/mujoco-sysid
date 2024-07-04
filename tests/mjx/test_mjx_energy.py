import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from mujoco_sysid import regressors as mj_reg
from mujoco_sysid.mjx import parameters, regressors

key = jax.random.PRNGKey(0)


def test_energy_match_z1():
    from robot_descriptions.z1_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # alter the model so it becomes mjx compatible
    mjmodel.dof_frictionloss = 0
    mjmodel.opt.integrator = 0
    # disable friction, contact and limits
    mjmodel.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    mjdata = mujoco.MjData(mjmodel)

    mjxmodel = mjx.put_model(mjmodel)
    assert len(mjxmodel.jnt_bodyid) == mjmodel.njnt, f"{len(mjxmodel.jnt_bodyid)} != {mjmodel.njnt}"
    # mjxdata = mjx.put_data(mjmodel, mjdata)

    # enable energy
    mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    # get vector of dynamic parameters
    theta = jnp.concatenate([parameters.get_dynamic_parameters(mjxmodel, i) for i in mjxmodel.jnt_bodyid])
    assert theta.shape == (mjxmodel.njnt * 10,)

    # generate random samples of configuration and velocity
    N_SAMPLES = 10000
    sampled_q, sampled_v = (
        jax.random.normal(key, (mjmodel.nq, N_SAMPLES)),
        jax.random.normal(key, (mjmodel.nv, N_SAMPLES)),
    )

    # compute the expected energy for each sample
    expected_energy = []
    for i in range(N_SAMPLES):
        mjdata.qpos[:] = sampled_q[:, i]
        mjdata.qvel[:] = sampled_v[:, i]

        mujoco.mj_step(mjmodel, mjdata)

        mj_en = mjdata.energy.copy()
        mj_en[0] += mj_reg.potential_energy_bias(mjmodel)

        expected_energy.append(jnp.sum(mj_en))

    expected_energy = jnp.array(expected_energy)
    # compute the batched regressors using mjx and find the computed energy

    @jax.vmap
    def compute_energy(q, v):
        mjx_data = mjx.make_data(mjxmodel)
        mjx_data = mjx_data.replace(qpos=q, qvel=v)
        mjx_data = mjx.step(mjxmodel, mjx_data)

        # compute the regressor
        reg_en = regressors.energy_regressor(mjxmodel, mjx_data)[2]
        computed_energy = jnp.dot(reg_en, theta)

        return computed_energy

    # compute the energy for each sample
    computed_energy = jax.jit(compute_energy)(sampled_q.T, sampled_v.T)

    # compare the computed energy with the expected energy
    assert jnp.allclose(expected_energy, computed_energy, atol=1e-5)


def test_energy_match_skydio():
    from robot_descriptions.skydio_x2_mj_description import MJCF_PATH

    mjmodel = mujoco.MjModel.from_xml_path(MJCF_PATH)
    # alter the model so it becomes mjx compatible
    mjmodel.dof_frictionloss = 0
    mjmodel.opt.integrator = 0
    # disable friction, contact and limits
    mjmodel.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    mjdata = mujoco.MjData(mjmodel)

    mjxmodel = mjx.put_model(mjmodel)
    assert len(mjxmodel.jnt_bodyid) == mjmodel.njnt, f"{len(mjxmodel.jnt_bodyid)} != {mjmodel.njnt}"
    # mjxdata = mjx.put_data(mjmodel, mjdata)

    # enable energy
    mjmodel.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    # get vector of dynamic parameters
    theta = jnp.concatenate([parameters.get_dynamic_parameters(mjxmodel, i) for i in mjxmodel.jnt_bodyid])
    assert theta.shape == (mjxmodel.njnt * 10,)

    # generate random samples of configuration and velocity
    N_SAMPLES = 10000
    sampled_q, sampled_v = (
        jax.random.normal(key, (mjmodel.nq, N_SAMPLES)),
        jax.random.normal(key, (mjmodel.nv, N_SAMPLES)),
    )

    # compute the expected energy for each sample
    expected_energy = []
    for i in range(N_SAMPLES):
        mjdata.qpos[:] = sampled_q[:, i]
        mjdata.qvel[:] = sampled_v[:, i]

        mujoco.mj_step(mjmodel, mjdata)

        mj_en = mjdata.energy.copy()
        mj_en[0] += mj_reg.potential_energy_bias(mjmodel)

        expected_energy.append(jnp.sum(mj_en))

    expected_energy = jnp.array(expected_energy)
    # compute the batched regressors using mjx and find the computed energy

    @jax.vmap
    def compute_energy(q, v):
        mjx_data = mjx.make_data(mjxmodel)
        mjx_data = mjx_data.replace(qpos=q, qvel=v)
        mjx_data = mjx.step(mjxmodel, mjx_data)

        # compute the regressor
        reg_en = regressors.energy_regressor(mjxmodel, mjx_data)
        computed_energy = jnp.dot(reg_en, theta)

        return computed_energy

    # compute the energy for each sample
    computed_energy = jax.jit(compute_energy)(sampled_q.T, sampled_v.T)

    # compare the computed energy with the expected energy
    assert jnp.allclose(expected_energy, computed_energy, atol=1e-5)


if __name__ == "__main__":
    test_energy_match_z1()
    test_energy_match_skydio()
