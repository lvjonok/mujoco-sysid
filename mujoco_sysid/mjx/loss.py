import jax.numpy as jnp
from jax import lax


def create_compute_loss(weight_matrix, forgetting_factor, x_diff):
    """
    Create a function to compute the loss between the predicted state and the measured state.

    Args:
        weight_matrix (jnp.ArrayLike): A matrix of weights to apply via matrix multiplication.
        forgetting_factor (float): A forgetting factor for the running sum.
        x_diff (callable): A function to compute the difference between x_hat and x_measured.

    Returns:
        callable: A function that takes x_hat and x_measured as input and returns the computed loss.
    """

    def compute_loss(x_hat, x_measured):
        # Define a function to compute the loss for each time step
        def step(carry, t):
            # Carry is the accumulated loss
            accumulated_loss = carry

            # Compute the difference at time step t using the provided x_diff function
            diff = x_diff(x_hat[t], x_measured[t])  # Assume diff is a vector

            # Apply the weight matrix to the difference (matrix multiplication)
            weighted_diff = weight_matrix @ diff

            # Compute the L2 norm of the weighted difference
            norm2_diff = jnp.linalg.norm(weighted_diff, 2)

            # Update the accumulated loss with the forgetting factor applied
            updated_loss = accumulated_loss + (forgetting_factor**t) * norm2_diff

            return updated_loss, updated_loss

        # Initialize loss to zero
        initial_loss = 0.0

        # Use jax.lax.scan to iterate over time steps and compute the loss
        final_loss, _ = lax.scan(step, initial_loss, jnp.arange(len(x_hat)))

        return final_loss

    return compute_loss


# def compute_loss(x_hat, x_measured, weight_matrix, forgetting_factor, x_diff):
#     """Compute the loss between the predicted state and the measured state

#     Args:
#         x_hat (jnp.ArrayLike): predicted state
#         x_measured (jnp.ArrayLike): measured state
#         weight_matrix (jnp.ArrayLike): matrix of weights to apply via matrix multiplication
#         forgetting_factor (float): forgetting factor for the running sum
#         x_diff (callable): a function to compute the difference between x_hat and x_measured

#     Returns:
#         jnp.ArrayLike: loss
#     """

#     # Define a function to compute the loss for each time step
#     def step(carry, t):
#         # Carry is the accumulated loss
#         accumulated_loss = carry

#         # Compute the difference at time step t using the provided x_diff function
#         diff = x_diff(x_hat[t], x_measured[t])  # Assume diff is a vector

#         # Apply the weight matrix to the difference (matrix multiplication)
#         weighted_diff = weight_matrix @ diff

#         # Compute the L2 norm of the weighted difference
#         norm2_diff = jnp.linalg.norm(weighted_diff, 2)

#         # Update the accumulated loss with the forgetting factor applied
#         updated_loss = accumulated_loss + (forgetting_factor**t) * norm2_diff

#         return updated_loss, updated_loss

#     # Initialize loss to zero
#     initial_loss = 0.0

#     # Use jax.lax.scan to iterate over time steps and compute the loss
#     final_loss, _ = lax.scan(step, initial_loss, jnp.arange(len(x_hat)))

#     return final_loss
