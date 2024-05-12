"""
This module provides functions for converting between different representations of pseudo inertia matrices,
including theta parameters, logarithmic Cholesky parameters, and Cholesky decomposition.

The main representations used are:
- Theta parameters: A 10-dimensional vector containing mass, first moments, and inertia tensor components.
    theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]

- Pseudo inertia matrix: A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.
    pseudo = [[-0.5*I_xx + 0.5*I_yy + 0.5*I_zz, -I_xy, -I_xz, mr_x],
              [-I_xy, 0.5*I_xx - 0.5*I_yy + 0.5*I_zz, -I_yz, mr_y],
              [-I_xz, -I_yz, 0.5*I_xx + 0.5*I_yy - 0.5*I_zz, mr_z],
              [mr_x, mr_y, mr_z, m]]

- Cholesky decomposition: An upper triangular matrix U such that the pseudo inertia matrix J = U * U^T.
    U = [[exp(alpha)*exp(d_1), s_12*exp(alpha), s_13*exp(alpha), t_1*exp(alpha)],
         [0, exp(alpha)*exp(d_2), s_23*exp(alpha), t_2*exp(alpha)],
         [0, 0, exp(alpha)*exp(d_3), t_3*exp(alpha)],
         [0, 0, 0, exp(alpha)]]

- Logarithmic Cholesky parameters: A 10-dimensional vector containing logarithmic parameters derived from the Cholesky decomposition.
    logchol = [alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2, t_3]

Functions:
- `theta2pseudo(theta: np.ndarray) -> np.ndarray`
    Converts theta parameters to a pseudo inertia matrix.
- `pseudo2theta(pseudo_inertia: np.ndarray) -> np.ndarray`
    Converts a pseudo inertia matrix to theta parameters.
- `logchol2chol(log_cholesky: np.ndarray) -> np.ndarray`
    Converts logarithmic Cholesky parameters to the Cholesky matrix, factoring out exp(alpha).
- `chol2logchol(U: np.ndarray) -> np.ndarray`
    Converts the upper triangular matrix U to logarithmic Cholesky parameters.
- `pseudo2cholesky(pseudo_inertia: np.ndarray) -> np.ndarray`
    Computes the Cholesky decomposition of a pseudo inertia matrix.
- `cholesky2pseudo(U: np.ndarray) -> np.ndarray`
    Converts the upper triangular Cholesky matrix U back into a pseudo inertia matrix.
- `pseudo2logchol(pseudo_inertia: np.ndarray) -> np.ndarray`
    Converts a pseudo inertia matrix to logarithmic Cholesky parameters.
- `theta2logchol(theta: np.ndarray) -> np.ndarray`
    Converts theta parameters directly to logarithmic Cholesky parameters.

For more information and derivations, please consider reviewing the following references:
- Rucker C, Wensing PM. Smooth parameterization of rigid-body inertia. IEEE Robotics and Automation Letters. 2022 Jan 21;7(2):2771-8.
- Wensing PM, Kim S, Slotine JJ. Linear matrix inequalities for physically consistent inertial parameter identification: 
        A statistical perspective on the mass distribution. IEEE Robotics and Automation Letters. 2017 Jul 20;3(1):60-7.

Additional discussions and derivations are available at:
https://colab.research.google.com/drive/1xFte2FT0nQ0ePs02BoOx4CmLLw5U-OUZ#scrollTo=Xt86l6AtZBhI
"""

import numpy as np


def theta2pseudo(theta: np.ndarray) -> np.ndarray:
    """
    Converts theta parameters to a pseudo inertia matrix.

    Args:
        theta (np.ndarray): A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)

    Returns:
        np.ndarray: A 4x4 pseudo inertia matrix.
    """
    m = theta[0]
    h = theta[1:4]
    I_xx, I_xy, I_yy, I_xz, I_yz, I_zz = theta[4:]

    I_bar = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    Sigma = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar

    pseudo_inertia = np.zeros((4, 4))
    pseudo_inertia[:3, :3] = Sigma
    pseudo_inertia[:3, 3] = h
    pseudo_inertia[3, :3] = h
    pseudo_inertia[3, 3] = m

    return pseudo_inertia


def pseudo2theta(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Converts a pseudo inertia matrix to theta parameters.

    Args:
        pseudo_inertia (np.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        np.ndarray: A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)
    """
    m = pseudo_inertia[3, 3]
    h = pseudo_inertia[:3, 3]
    Sigma = pseudo_inertia[:3, :3]

    I_bar = np.trace(Sigma) * np.eye(3) - Sigma

    I_xx = I_bar[0, 0]
    I_xy = I_bar[0, 1]
    I_yy = I_bar[1, 1]
    I_xz = I_bar[0, 2]
    I_yz = I_bar[1, 2]
    I_zz = I_bar[2, 2]

    theta = np.array([m, h[0], h[1], h[2], I_xx, I_xy, I_yy, I_xz, I_yz, I_zz])

    return theta


def logchol2chol(log_cholesky):
    """
    Converts logarithmic Cholesky parameters to the Cholesky matrix, factoring out exp(alpha).

    Args:
        log_cholesky (np.ndarray): A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d1, d2, d3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t1, t2, t3)

    Returns:
        np.ndarray: A 4x4 upper triangular Cholesky matrix U.
    """
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    # Compute the exponential terms
    exp_alpha = np.exp(alpha)
    exp_d1 = np.exp(d1)
    exp_d2 = np.exp(d2)
    exp_d3 = np.exp(d3)

    # Construct the scaled Cholesky matrix U without exp_alpha
    U = np.zeros((4, 4))
    U[0, 0] = exp_d1
    U[0, 1] = s12
    U[0, 2] = s13
    U[0, 3] = t1
    U[1, 1] = exp_d2
    U[1, 2] = s23
    U[1, 3] = t2
    U[2, 2] = exp_d3
    U[2, 3] = t3
    U[3, 3] = 1

    # Multiply the entire matrix by exp_alpha
    U *= exp_alpha

    return U


def chol2logchol(U: np.ndarray) -> np.ndarray:
    """
    Converts the upper triangular matrix U to logarithmic Cholesky parameters.

    Args:
        U (np.ndarray): A 4x4 upper triangular matrix decomposition of the pseudo inertia matrix.

    Returns:
        np.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """

    alpha = np.log(U[3, 3])
    d1 = np.log(U[0, 0] / U[3, 3])
    d2 = np.log(U[1, 1] / U[3, 3])
    d3 = np.log(U[2, 2] / U[3, 3])
    s12 = U[0, 1] / U[3, 3]
    s23 = U[1, 2] / U[3, 3]
    s13 = U[0, 2] / U[3, 3]
    t1 = U[0, 3] / U[3, 3]
    t2 = U[1, 3] / U[3, 3]
    t3 = U[2, 3] / U[3, 3]
    return np.array([alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3])


def logchol2theta(log_cholesky: np.ndarray) -> np.ndarray:
    """
    Converts logarithmic Cholesky parameters directly to theta parameters.

    Args:
        log_cholesky (np.ndarray): A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s12, s23, s13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)

    Returns:
        np.ndarray: A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)
    """
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    # Calculate exponential terms without applying exp_2_alpha
    exp_d1 = np.exp(d1)
    exp_d2 = np.exp(d2)
    exp_d3 = np.exp(d3)

    # Compute the elements of the output vector without exp_2_alpha
    theta = np.zeros(10)
    theta[0] = 1
    theta[1] = t1
    theta[2] = t2
    theta[3] = t3
    theta[4] = s23**2 + t2**2 + t3**2 + exp_d2**2 + exp_d3**2
    theta[5] = -s12 * exp_d2 - s13 * s23 - t1 * t2
    theta[6] = s12**2 + s13**2 + t1**2 + t3**2 + exp_d1**2 + exp_d3**2
    theta[7] = -s13 * exp_d3 - t1 * t3
    theta[8] = -s23 * exp_d3 - t2 * t3
    theta[9] = s12**2 + s13**2 + s23**2 + t1**2 + t2**2 + exp_d1**2 + exp_d2**2

    # Calculate exp_2_alpha and scale the theta vector
    exp_2_alpha = np.exp(2 * alpha)
    theta *= exp_2_alpha

    return theta


def pseudo2cholesky(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Computes the Cholesky decomposition of a pseudo inertia matrix.
    Note that this is UPPER triangular decomposition in form J = U*U^T, which is not the usual way to calculate the Cholesky decomposition.
    However, in this form, the associated logarithmic Cholesky parameters have a geometrical meaning.

    Args:
        pseudo_inertia (np.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        np.ndarray: A 4x4 upper triangular Cholesky matrix U.
    """

    n = pseudo_inertia.shape[0]
    indices = np.arange(n - 1, -1, -1)  # Indices to reverse the order

    # Apply the inversion using indices for rows and columns
    reversed_inertia = pseudo_inertia[indices][:, indices]

    # Perform Cholesky decomposition on the permuted matrix A'
    L_prime = np.linalg.cholesky(reversed_inertia)

    # Apply the reverse permutation to L_prime and transpose it to form U
    U = L_prime[indices][:, indices]

    return U


def cholesky2pseudo(U: np.ndarray) -> np.ndarray:
    """
    Converts the upper triangular Cholesky matrix U back into a pseudo inertia matrix.

    Args:
        U (np.ndarray): A 4x4 upper triangular Cholesky matrix.

    Returns:
        np.ndarray: A 4x4 pseudo inertia matrix.
    """
    return U @ U.T


def pseudo2logchol(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Converts a pseudo inertia matrix to logarithmic Cholesky parameters.

    Args:
        pseudo_inertia (np.ndarray): A 4x4 symmetric matrix representing the pseudo inertia of a rigid body.

    Returns:
        np.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s_12, s_23, s_13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """
    # theta = pseudo2theta(pseudo_inertia)
    U = pseudo2cholesky(pseudo_inertia)

    logchol = chol2logchol(U)

    return logchol


def theta2logchol(theta: np.ndarray) -> np.ndarray:
    """
    Converts theta parameters directly to logarithmic Cholesky parameters.

    Args:
        theta (np.ndarray): A 10-dimensional vector containing mass, first moments, and inertia tensor components.
            - theta[0]: Mass (m)
            - theta[1:4]: First moments (mc_x, mc_y, mc_z)
            - theta[4:10]: Inertia tensor components (I_xx, I_yy, I_zz, I_xy, I_xz, I_yz)

    Returns:
        np.ndarray: A 10-dimensional vector containing logarithmic Cholesky parameters.
            - log_cholesky[0]: alpha
            - log_cholesky[1:4]: (d_1, d_2, d_3)
            - log_cholesky[4:7]: (s_12, s_23, s_13)
            - log_cholesky[7:10]: (t_1, t_2, t_3)
    """
    pseudo_inertia = theta2pseudo(theta)
    return pseudo2logchol(pseudo_inertia)
