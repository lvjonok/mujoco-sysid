import jax.numpy as np


def theta2pseudo(theta: np.ndarray) -> np.ndarray:
    m = theta[0]
    h = theta[1:4]
    I_xx, I_xy, I_yy, I_xz, I_yz, I_zz = theta[4:]

    I_bar = np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])

    Sigma = 0.5 * np.trace(I_bar) * np.eye(3) - I_bar

    pseudo_inertia = np.zeros((4, 4))
    pseudo_inertia = pseudo_inertia.at[:3, :3].set(Sigma)
    pseudo_inertia = pseudo_inertia.at[:3, 3].set(h)
    pseudo_inertia = pseudo_inertia.at[3, :3].set(h)
    pseudo_inertia = pseudo_inertia.at[3, 3].set(m)

    return pseudo_inertia


def pseudo2theta(pseudo_inertia: np.ndarray) -> np.ndarray:
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
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    exp_alpha = np.exp(alpha)
    exp_d1 = np.exp(d1)
    exp_d2 = np.exp(d2)
    exp_d3 = np.exp(d3)

    U = np.zeros((4, 4))
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

    U *= exp_alpha

    return U


def chol2logchol(U: np.ndarray) -> np.ndarray:
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
    alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3 = log_cholesky

    exp_d1 = np.exp(d1)
    exp_d2 = np.exp(d2)
    exp_d3 = np.exp(d3)

    theta = np.array(
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

    exp_2_alpha = np.exp(2 * alpha)
    theta *= exp_2_alpha

    return theta


def pseudo2cholesky(pseudo_inertia: np.ndarray) -> np.ndarray:
    n = pseudo_inertia.shape[0]
    indices = np.arange(n - 1, -1, -1)

    reversed_inertia = pseudo_inertia[indices][:, indices]

    L_prime = np.linalg.cholesky(reversed_inertia)

    U = L_prime[indices][:, indices]

    return U


def cholesky2pseudo(U: np.ndarray) -> np.ndarray:
    return U @ U.T


def pseudo2logchol(pseudo_inertia: np.ndarray) -> np.ndarray:
    U = pseudo2cholesky(pseudo_inertia)
    logchol = chol2logchol(U)
    return logchol


def theta2logchol(theta: np.ndarray) -> np.ndarray:
    pseudo_inertia = theta2pseudo(theta)
    return pseudo2logchol(pseudo_inertia)
