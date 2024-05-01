# MuJoCo SysId

<a href="https://colab.research.google.com/drive/1WXzbk1fTAikywImj6VpcZ9BMdPC7Fn4n#scrollTo=g-cUBZu4hnyE"><img src="https://colab.research.google.com/assets/colab-badge.svg" width="120" align="center"/></a>

# <h1><center>System Identification in Robotic Systems<br></center></h1>

This repository offers a concise introduction to system identification (Sys-ID) in robotic systems. It is inspired by recent advancements in MuJoCo utilities, specifically the [Levenberg-Marquardt nonlinear least squares method](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb).

Our primary focus is on mechanical systems where **the model structure is known**, including the number of state variables and the configuration of the kinematic tree. While the dynamics of these systems can be inherently complex, the general forms of the equations are known and have already been implemented in MuJoCo. In this context, the task of identification is essentially to estimate the parameters within a structured model.

### Contents

This repository includes the following examples:

- [Estimation of cart-pole inertial parameters through random forcing and LQR stabilization of the identified system](examples/cart_pole.ipynb)
- [Identification of end-effector load for the Franka Emika Panda and compensation using inverse dynamics](examples/panda.ipynb)
- [Determination of mass, center of mass, and spatial inertia for a Skydio X2 Quadrotor following LTV LQR tracking attempts.](examples/skydio.ipynb)

Additionally, we provide some theoretical background and introduce two utility functions that may enhance system identification and adaptive control in robotic systems: `mj_bodyRegressor` and `mj_jointRegressor`.

We hope these examples and utilities will be useful for all MuJoCo users and assist in resolving their system identification challenges.

For further questions and suggestions, please do not hesitate to create an issue or start a discussion [on GitHub](https://github.com/lvjonok/mujoco-sysid/issues/new/choose).
