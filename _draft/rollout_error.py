@jax.jit
def rollout_errors(parameters, states, controls):
    # TODO: Use the full trajecttory in shouting not only las point
    interval_initial_states = states[::HORIZON]
    interval_controls = jnp.reshape(controls, (N_INTERVALS, HORIZON))
    batched_rollout = jax.vmap(rollout_trajectory, in_axes=(None, None, 0, 0))
    predicted_states_trajectories = batched_rollout(parameters, mjx_model, interval_initial_states, interval_controls)
    interval_states_trajectories = jnp.reshape(states, jnp.shape(predicted_states_trajectories))
    loss = jnp.mean(
        optax.l2_loss(predicted_states_trajectories, interval_states_trajectories)
    )  # + 0.05*jnp.mean(optax.huber_loss(parameters, jnp.zeros_like(parameters)))
    return loss
