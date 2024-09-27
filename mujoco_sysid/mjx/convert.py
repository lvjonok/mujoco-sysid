import jax.numpy as jnp


def theta2pseudo(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Converts theta parameters to a pseudo inertia matrix.

    Args:
        theta (jnp.ndarray): A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)

    Returns:
        jnp.ndarray: A 4x4 pseudo inertia matrix.
    """
    m = theta[0]
    h = theta[1:4]
    I_xx, I_xy, I_yy, I_xz, I_yz, I_zz = theta[4:]

    I_bar = jnp.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    Sigma = 0.5 * jnp.trace(I_bar) * jnp.eye(3) - I_bar

    pseudo_inertia = jnp.zeros((4, 4))
    pseudo_inertia = pseudo_inertia.at[:3, :3].set(Sigma)
    pseudo_inertia = pseudo_inertia.at[:3, 3].set(h)
    pseudo_inertia = pseudo_inertia.at[3, :3].set(h)
    pseudo_inertia = pseudo_inertia.at[3, 3].set(m)

    return pseudo_inertia


def pseudo2theta(pseudo_inertia: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a pseudo inertia matrix to theta parameters.

    Args:
        pseudo_inertia (jnp.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        jnp.ndarray: A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)
    """
    m = pseudo_inertia[3, 3]
    h = pseudo_inertia[:3, 3]
    Sigma = pseudo_inertia[:3, :3]

    I_bar = jnp.trace(Sigma) * jnp.eye(3) - Sigma

    I_xx = I_bar[0, 0]
    I_xy = I_bar[0, 1]
    I_yy = I_bar[1, 1]
    I_xz = I_bar[0, 2]
    I_yz = I_bar[1, 2]
    I_zz = I_bar[2, 2]

    theta = jnp.array([m, h[0], h[1], h[2], I_xx, I_xy, I_yy, I_xz, I_yz, I_zz])

    return theta


def logchol2chol(log_cholesky):
    """
    Converts logarithmic Cholesky parameters to the Cholesky matrix, factoring out exp(alpha).

    Args:
        log_cholesky (jnp.ndarray): A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d1, d2, d3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t1, t2, t3)

    Returns:
        jnp.ndarray: A 4x4 upper triangular Cholesky matrix U.
    """
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    # Compute the exponential terms
    exp_alpha = jnp.exp(alpha)
    exp_d1 = jnp.exp(d1)
    exp_d2 = jnp.exp(d2)
    exp_d3 = jnp.exp(d3)

    # Construct the scaled Cholesky matrix U without exp_alpha
    U = jnp.zeros((4, 4))
    U = U.at[0, 0].set(exp_d1)
    U = U.at[0, 1].set(s12)
    U = U.at[0, 2].set(s13)
    U = U.at[0, 3].set(t1)
    U = U.at[1, 1].set(exp_d2)
    U = U.at[1, 2].set(s23)
    U = U.at[1, 3].set(t2)
    U = U.at[2, 2].set(exp_d3)
    U = U.at[2, 3].set(t3)
    U = U.at[3, 3].set(1)

    # Multiply the entire matrix by exp_alpha
    U *= exp_alpha

    return U


def chol2logchol(U: jnp.ndarray) -> jnp.ndarray:
    """
    Converts the upper triangular matrix U to logarithmic Cholesky parameters.

    Args:
        U (jnp.ndarray): A 4x4 upper triangular matrix decomposition of the pseudo inertia matrix.

    Returns:
        jnp.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """

    alpha = jnp.log(U[3, 3])
    d1 = jnp.log(U[0, 0] / U[3, 3])
    d2 = jnp.log(U[1, 1] / U[3, 3])
    d3 = jnp.log(U[2, 2] / U[3, 3])
    s12 = U[0, 1] / U[3, 3]
    s23 = U[1, 2] / U[3, 3]
    s13 = U[0, 2] / U[3, 3]
    t1 = U[0, 3] / U[3, 3]
    t2 = U[1, 3] / U[3, 3]
    t3 = U[2, 3] / U[3, 3]
    return jnp.array([alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3])


def logchol2theta(log_cholesky: jnp.ndarray) -> jnp.ndarray:
    """
    Converts logarithmic Cholesky parameters directly to theta parameters.

    Args:
        log_cholesky (jnp.ndarray): A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)

    Returns:
        jnp.ndarray: A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)
    """
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    # Calculate exponential terms without applying exp_2_alpha
    exp_d1 = jnp.exp(d1)
    exp_d2 = jnp.exp(d2)
    exp_d3 = jnp.exp(d3)

    # Compute the elements of the output vector without exp_2_alpha
    theta = jnp.array(
        [
            1,
            t1,
            t2,
            t3,
            s23**2 + t2**2 + t3**2 + exp_d2**2 + exp_d3**2,
            -s12 * exp_d2 - s13 * s23 - t1 * t2,
            s12**2 + s13**2 + t1**2 + t3**2 + exp_d1**2 + exp_d3**2,
            -s13 * exp_d3 - t1 * t3,
            -s23 * exp_d3 - t2 * t3,
            s12**2 + s13**2 + s23**2 + t1**2 + t2**2 + exp_d1**2 + exp_d2**2,
        ]
    )

    # Calculate exp_2_alpha and scale the theta vector
    exp_2_alpha = jnp.exp(2 * alpha)
    theta *= exp_2_alpha

    return theta


def pseudo2cholesky(pseudo_inertia: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Cholesky decomposition of a pseudo inertia matrix.
    Note that this is UPPER triangular decomposition in form J = U*U^T, which is not the usual way to calculate the Cholesky decomposition.
    However, in this form, the associated logarithmic Cholesky parameters have a geometrical meaning.

    Args:
        pseudo_inertia (jnp.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        jnp.ndarray: A 4x4 upper triangular Cholesky matrix U.
    """  # noqa: E501

    n = pseudo_inertia.shape[0]
    indices = jnp.arange(n - 1, -1, -1)  # Indices to reverse the order

    # Apply the inversion using indices for rows and columns
    reversed_inertia = pseudo_inertia[indices][:, indices]

    # Perform Cholesky decomposition on the permuted matrix A'
    L_prime = jnp.linalg.cholesky(reversed_inertia)

    # Apply the reverse permutation to L_prime and transpose it to form U
    U = L_prime[indices][:, indices]

    return U


def cholesky2pseudo(U: jnp.ndarray) -> jnp.ndarray:
    """
    Converts the upper triangular Cholesky matrix U back into a pseudo inertia matrix.

    Args:
        U (jnp.ndarray): A 4x4 upper triangular Cholesky matrix.

    Returns:
        jnp.ndarray: A 4x4 pseudo inertia matrix.
    """
    return U @ U.T


def pseudo2logchol(pseudo_inertia: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a pseudo inertia matrix to logarithmic Cholesky parameters.

    Args:
        pseudo_inertia (jnp.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        jnp.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s_12, s_23, s_13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """
    # theta = pseudo2theta(pseudo_inertia)
    U = pseudo2cholesky(pseudo_inertia)

    logchol = chol2logchol(U)

    return logchol


def theta2logchol(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Converts theta parameters directly to logarithmic Cholesky parameters.

    Args:
        theta (jnp.ndarray): A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)

    Returns:
        jnp.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s_12, s_23, s_13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """
    pseudo_inertia = theta2pseudo(theta)
    return pseudo2logchol(pseudo_inertia)
