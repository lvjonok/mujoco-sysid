from typing import Callable

import jax
import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx

from .parameters import set_dynamic_parameters


def step(
    model: mjx.Model,
    x: jpt.ArrayLike,
    ctrl: jpt.ArrayLike,
) -> tuple[jpt.ArrayLike, jpt.ArrayLike]:
    """Simple step function for the simulation:

    Args:
        model (mjx.Model): MJX model
        x (jpt.ArrayLike): current state
        ctrl (jpt.ArrayLike): control input

    Returns:
        tuple[jpt.ArrayLike, jpt.ArrayLike]: updated configuration and velocity
    """
    data = mjx.make_data(model).replace(qpos=x[: model.nq], qvel=x[model.nq :], ctrl=ctrl)
    data = mjx.step(model, data)

    return jnp.concatenate([data.qpos, data.qvel])


def parameters_map(parameters: jnp.ndarray, model: mjx.Model) -> mjx.Model:
    """
    Map new parameters to the model.

    Args:
        parameters (jnp.ndarray): Array of parameters to be mapped.
        model (mjx.Model): The original MJX model.

    Returns:
        mjx.Model: Updated model with new parameters.
    """
    # Assuming parameters are for all bodies except the world body
    n_bodies = len(model.body_mass) - 1
    per_body = jnp.split(parameters, n_bodies)

    for i in range(n_bodies):
        model = set_dynamic_parameters(model, i + 1, per_body[i])

    return model


def create_rollout(parameters_map: Callable, step: Callable = step) -> Callable:
    """Create a closure for rolling out a trajectory.

    Args:
        parameters_map (Callable): (parameters, mjx_model) -> mjx_model
        step (Callable, optional): (mjx_model, state, control). Defaults to step.

    Returns:
        Callable: _description_
    """

    def rollout_trajectory(
        parameters: jnp.ndarray,
        model: mjx.Model,
        initial_state: jnp.ndarray,
        control_inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Rollout a trajectory given parameters, initial state, and control inputs.

        Args:
            parameters (jnp.ndarray): Parameters for the model.
            parameters_map (Callable): Function to map parameters to the model.
            model (mjx.Model): The original MJX model.
            initial_state (jnp.ndarray): Initial state of the system.
            control_inputs (jnp.ndarray): Control inputs for each step.

        Returns:
            jnp.ndarray: States of the system for each step.
        """

        def step_fn(state, control):
            new_state = step(model, state, control)
            return new_state, new_state

        model = parameters_map(parameters, model)
        (_, states) = jax.lax.scan(step_fn, initial_state, control_inputs)
        return states

    return rollout_trajectory
