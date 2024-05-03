"""
This module provides a suite of functions designed to work with pseudo inertia matrices and their various transformations. 
It supports converting inertial parameters, denoted as `theta = [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]`, 
into pseudo inertia matrices and back, as well as conversions involving log Cholesky parameterizations. 
These functions facilitate system identification by accounting for the manifold of PD (Positive Definite) 
pseudo inertia and can be used alongside the nonlinear optimization tools provided by the Mujoco minimizer.

### Functions:
- `theta2pseudo`: Converts theta parameters into pseudo inertia matrices.
- `pseudo2theta`: Converts pseudo inertia matrices back into theta parameters.
- `pseudo2cholesky`: Computes the Cholesky decomposition of a pseudo inertia matrix.
- `logchol2theta`: Converts log Cholesky parameters back into theta parameters.
- `chol2logchol`: Converts a Cholesky decomposition into log Cholesky parameters.
- `pseudo2logchol`: Converts a pseudo inertia matrix into log Cholesky parameters.
- `theta2logchol`: Converts theta parameters directly into log Cholesky parameters.

For more information, please consider reviewing the following references:
- Rucker C, Wensing PM. Smooth parameterization of rigid-body inertia. IEEE Robotics and Automation Letters. 2022 Jan 21;7(2):2771-8.
- Wensing PM, Kim S, Slotine JJ. Linear matrix inequalities for physically consistent inertial parameter identification: 
        A statistical perspective on the mass distribution. IEEE Robotics and Automation Letters. 2017 Jul 20;3(1):60-7.
"""

import numpy as np
from scipy.linalg import cholesky


def theta2pseudo(theta: np.ndarray) -> np.ndarray:
    """
    Converts theta parameters into the pseudo inertia matrix.

    Args:
        theta (np.ndarray): Contains mass, first moments, and inertia tensor components:
            [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
    Returns:
        np.ndarray: Pseudo inertia matrix.
    """
    m = theta[0]
    h = theta[1:4]
    I_xx, I_xy, I_yy, I_xz, I_yz, I_zz = theta[4:]

    I_bar = np.array([[I_xx, I_xy, I_xz],
                      [I_xy, I_yy, I_yz],
                      [I_xz, I_yz, I_zz]])

    Sigma = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar

    pseudo_inertia = np.zeros((4, 4))
    pseudo_inertia[:3, :3] = Sigma
    pseudo_inertia[:3, 3] = h
    pseudo_inertia[3, :3] = h
    pseudo_inertia[3, 3] = m

    return pseudo_inertia


def pseudo2theta(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Converts a pseudo inertia matrix back to theta parameters.

    Args:
        pseudo_inertia (np.ndarray): Pseudo inertia matrix.

    Returns:
        np.ndarray: Vector of parameters [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz].
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


def pseudo2cholesky(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Computes the Cholesky decomposition of a pseudo inertia matrix.

    Args:
        pseudo_inertia (np.ndarray): Pseudo inertia matrix.

    Returns:
        np.ndarray: Cholesky decomposition of the matrix.
    """
    return cholesky(pseudo_inertia)


def logchol2theta(log_cholesky: np.ndarray) -> np.ndarray:
    """
    Converts logarithmic Cholesky parameters to theta parameters.

    Args:
        log_cholesky (np.ndarray): Logarithmic Cholesky parameters.

    Returns:
        np.ndarray: Vector of parameters [m, h_x, h_y, h_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz].
    """
    alpha, d1, d2, d3, s12, s23, s34 = log_cholesky
    exp_diag = np.exp(np.array([d1, d2, d3]))
    m = exp_diag[2]**2
    h = np.array([alpha * exp_diag[0], s12 * exp_diag[1], s23 * exp_diag[2]])

    I_xx = exp_diag[0]**2
    I_xy = s12 * exp_diag[0] * exp_diag[1]
    I_yy = exp_diag[1]**2
    I_xz = s34 * exp_diag[0] * exp_diag[2]
    I_yz = s23 * exp_diag[1] * exp_diag[2]
    I_zz = exp_diag[2]**2

    theta = np.array([m, h[0], h[1], h[2], I_xx, I_xy, I_yy, I_xz, I_yz, I_zz])
    return theta


def chol2logchol(cholesky: np.ndarray) -> np.ndarray:
    """
    Converts Cholesky decomposition to logarithmic Cholesky parameters.

    Args:
        cholesky (np.ndarray): Cholesky decomposition of a matrix.

    Returns:
        np.ndarray: Logarithmic Cholesky parameters.
    """
    d1, d2, d3 = np.diag(cholesky)
    s12 = cholesky[0, 1] / d1
    s23 = cholesky[1, 2] / d2
    s34 = cholesky[0, 2] / d3
    log_cholesky = np.array(
        [np.log(d1), np.log(d2), np.log(d3), s12, s23, s34])
    return log_cholesky


def pseudo2logchol(pseudo_inertia: np.ndarray) -> np.ndarray:
    """
    Converts a pseudo inertia matrix to logarithmic Cholesky parameters.

    Args:
        pseudo_inertia (np.ndarray): Pseudo inertia matrix.

    Returns:
        np.ndarray: Logarithmic Cholesky parameters.
    """
    cholesky_decomp = cholesky(pseudo_inertia)
    return chol2logchol(cholesky_decomp)


def theta2logchol(theta: np.ndarray) -> np.ndarray:
    """
    Converts theta parameters directly to logarithmic Cholesky parameters.

    Args:
        theta (np.ndarray): Contains mass, first moments, and inertia tensor components.

    Returns:
        np.ndarray: Logarithmic Cholesky parameters.
    """
    pseudo_inertia = theta2pseudo(theta)
    return pseudo2logchol(pseudo_inertia)
