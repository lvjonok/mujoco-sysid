# MuJoCo SysId

MuJoCo System Identification (mujoco-sysid) is a Python module designed to perform system identification using the MuJoCo physics engine. This module facilitates the estimation of model parameters to match the simulated dynamics with observed data, enhancing the accuracy and reliability of robotic simulations.

# Features

- **Dynamic parameters**: getters and setters for dynamic parameters of MuJoCo and MJX models.
- **Regressor models**: energy and dynamics-based regressors for robotic systems.
- **Parameters representation**: utilities for converting between dynamic parameters, Log-Cholesky parametrization and pseudo-inertia.

# Installation

To install the module, run the following command:

```bash
pip install mujoco-sysid
```

In order to use with MJX use `[mjx]` or `[mjx_cpu]` for CPU-only version:

```bash
pip install mujoco-sysid[mjx]
```

# Documentation

To be generated.

# Examples

<!-- # <h1><center>System Identification in Robotic Systems<br></center></h1> -->

The demo is inspired by recent advancements in MuJoCo utilities, specifically the [Levenberg-Marquardt nonlinear least squares method](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb).

Our primary focus is on mechanical systems where **the model structure is known**, including the number of state variables and the configuration of the kinematic tree. While the dynamics of these systems can be inherently complex, the general forms of the equations are known and have already been implemented in MuJoCo. In this context, the task of identification is essentially to estimate the parameters within a structured model.

This repository includes the following [examples](examples/mujoco_sysid_demo.ipynb):
<a href="https://colab.research.google.com/github/lvjonok/mujoco-sysid/blob/master/examples/mujoco_sysid_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="120" align="center"/></a>

- Estimation of cart-pole inertial parameters through random forcing and LQR stabilization of the identified system
- Identification of end-effector load for the Franka Emika Panda and compensation using inverse dynamics
- Determination of mass, center of mass, and spatial inertia for a Skydio X2 Quadrotor following LTV LQR tracking attempts.

We hope these examples and utilities will be useful for all MuJoCo users and assist in resolving their system identification challenges.

For further questions and suggestions, please do not hesitate to create an issue or start a discussion [on GitHub](https://github.com/lvjonok/mujoco-sysid/issues/new/choose).
