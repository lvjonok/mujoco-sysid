import jax
import jax.numpy as jnp
import jax.typing as jpt
import mujoco.mjx as mjx

from .parameters import set_dynamic_parameters


def step(
    model: mjx.Model,
    x: jpt.ArrayLike,
    ctrl: jpt.ArrayLike,
    parameters: jpt.ArrayLike,
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
    # update the parameters in model
    # parameters should be split for each body
    per_body_parameters = jnp.split(parameters, len(model.body_mass) - 1)

    for i in range(len(model.body_mass) - 1):
        model = set_dynamic_parameters(model, i + 1, per_body_parameters[i])

    data = mjx.make_data(model).replace(qpos=x[: model.nq], qvel=x[model.nq :], ctrl=ctrl)
    data = mjx.step(model, data)

    return jnp.concatenate([data.qpos, data.qvel])


def rollout(
    model: mjx.Model,
    x0: jpt.ArrayLike,
    us: jpt.ArrayLike,
    parameters: jpt.ArrayLike,
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

    # Define a single step function to be used with lax.scan
    def step_fn(x, u):
        new_x = step(model, x, u, parameters)
        # Carry the next state and save the current state
        return new_x, new_x

    # Use lax.scan to roll over all control inputs in us
    (_, xs) = jax.lax.scan(step_fn, x0, us)

    return xs


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
