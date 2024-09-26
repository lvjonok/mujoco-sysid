import jax
import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx
from typing import Callable
from .parameters import set_dynamic_parameters


def step(
    parameters: jpt.ArrayLike,
    model: mjx.Model,
    x: jpt.ArrayLike,
    ctrl: jpt.ArrayLike,
) -> tuple[jpt.ArrayLike, jpt.ArrayLike]:
    """Smart step function that updates the parameters in the model

    Args:
        model (mjx.Model): MJX model
        x (jpt.ArrayLike): current state
        ctrl (jpt.ArrayLike): control input
        parameters (jpt.ArrayLike): 10 * nbodies parameters

    Returns:
        tuple[jpt.ArrayLike, jpt.ArrayLike]: updated configuration and velocity
    """
    model = set_dynamic_parameters(model, 1, parameters)

    data = mjx.make_data(model).replace(qpos=x[: model.nq], qvel=x[model.nq :], ctrl=ctrl)
    data = mjx.step(model, data)

    return jnp.concatenate([data.qpos, data.qvel])


def rollout(
    parameters: jpt.ArrayLike,
    model: mjx.Model,
    x0: jpt.ArrayLike,
    us: jpt.ArrayLike,
) -> jpt.ArrayLike:
    """Rollout the model with the given parameters using jax.lax.scan

    Args:
        model (mjx.Model): MJX model
        x0 (jpt.ArrayLike): initial state (nx,)
        us (jpt.ArrayLike): control inputs shape (N, nu)
        parameters (jpt.ArrayLike): 10 * nbodies parameters

    Returns:
        jpt.ArrayLike: states of the system shape (N, nx)
    """

    # update the parameters in model
    # parameters should be split for each body

    # per_body_parameters = jnp.split(parameters, len(model.body_mass) - 1)

    # for i in range(len(model.body_mass) - 1):
    #     model = set_dynamic_parameters(model, i + 1, per_body_parameters[i])
    model = set_dynamic_parameters(model, 1, parameters)

    # Define a single step function to be used with lax.scan
    def step_fn(x, u):
        new_x = step(parameters, model, x, u)
        # Carry the next state and save the current state
        return new_x, new_x

    # Use lax.scan to roll over all control inputs in us
    (_, xs) = jax.lax.scan(step_fn, x0, us)

    return xs


def rollout2(
    parameters: jpt.ArrayLike,
    model: mjx.Model,
    x0: jpt.ArrayLike,
    us: jpt.ArrayLike,
) -> jpt.ArrayLike:
    """Rollout the model with the given parameters using jax.lax.scan

    Args:
        model (mjx.Model): MJX model
        x0 (jpt.ArrayLike): initial state (nx,)
        us (jpt.ArrayLike): control inputs shape (N, nu)
        parameters (jpt.ArrayLike): 10 * nbodies parameters

    Returns:
        jpt.ArrayLike: states of the system shape (N, nx)
    """

    # update the parameters in model
    # parameters should be split for each body

    # per_body_parameters = jnp.split(parameters, len(model.body_mass) - 1)

    # for i in range(len(model.body_mass) - 1):
    #     model = set_dynamic_parameters(model, i + 1, per_body_parameters[i])
    model = set_dynamic_parameters(model, 1, parameters)

    x = x0
    xs = []
    for u in us:
        x = step(parameters, model, x, u)
        xs.append(x)

    return jnp.stack(xs)


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

    for i in range(n_bodies):
        model = set_dynamic_parameters(parameters, i + 1, model)

    return model


# @jax.jit(static_argnames=['parameters_map'])
def parametric_step(
    parameters: jnp.ndarray,
    model: mjx.Model,
    state: jnp.ndarray,
    control: jnp.ndarray,
    parameters_map: Callable,
) -> jnp.ndarray:
    """
    Perform a step with new parameter mapping.

    Args:
        parameters (jnp.ndarray): Parameters for the model.
        parameters_map (Callable): Function to map parameters to the model.
        model (mjx.Model): The original MJX model.
        state (jnp.ndarray): Current state.
        control (jnp.ndarray): Control input.

    Returns:
        jnp.ndarray: Updated state after the step.
    """
    new_model = parameters_map(parameters, model)

    data = mjx.make_data(new_model).replace(qpos=state[: new_model.nq], qvel=state[new_model.nq :], ctrl=control)
    data = mjx.step(new_model, data)

    return jnp.concatenate([data.qpos, data.qvel])


# @jax.jit(static_argnames=['parameters_map'])
def rollout_trajectory(
    parameters: jnp.ndarray,
    model: mjx.Model,
    initial_state: jnp.ndarray,
    control_inputs: jnp.ndarray,
    parameters_map: Callable,
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
        new_state = parametric_step(parameters, parameters_map, model, state, control)
        return new_state, new_state

    (_, states) = jax.lax.scan(step_fn, initial_state, control_inputs)
    return states


# class Problem:
#     def compute_loss(q, v, u, parameters):
#         # pass all the history to update the residuals
#         pass


# class RecursiveProblem:
#     def __init__(self, q, v, u, parameters, N) -> None:
#         # use sliding window
#         pass

#     # def compute


# class Residual:
#     def compute_residual(q, v, u, parameters):
#         pass
