import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Union

def setup_plot(nrows: int = 2, ncols: int = 2, figsize: Tuple[int, int] = (10, 10)) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Set up a matplotlib figure with the specified number of rows and columns.

    Args:
        nrows (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        figsize (Tuple[int, int]): Figure size in inches (width, height).

    Returns:
        Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]: Figure and axes objects.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    return fig, axes

def plot_state(ax: plt.Axes, timespan: np.ndarray, actual: np.ndarray, simulated: np.ndarray, label: str, color: str = 'blue', alpha: float = 0.5) -> None:
    """
    Plot actual and simulated state data on the given axes.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        timespan (np.ndarray): Array of time points.
        actual (np.ndarray): Array of actual state values.
        simulated (np.ndarray): Array of simulated state values.
        label (str): Label for the state (e.g., "Angle" or "Velocity").
        color (str): Color for the simulated data plot.
        alpha (float): Alpha value for the simulated data plot.
    """
    ax.plot(timespan, actual, label=f"Actual {label}", color="black", linestyle="dashed", linewidth=2)
    ax.plot(timespan, simulated, alpha=alpha, color=color, label=f"Simulated {label}")
    ax.set_ylabel(f"{label} (rad{'/' if label == 'Velocity' else ''}s)")
    ax.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.legend()

def plot_phase_portrait(ax: plt.Axes, angle: np.ndarray, velocity: np.ndarray, simulated_angle: np.ndarray, simulated_velocity: np.ndarray, color: str = 'blue', alpha: float = 0.5) -> None:
    """
    Plot the phase portrait of actual and simulated data.

    Args:
        ax (plt.Axes): The matplotlib axes to plot on.
        angle (np.ndarray): Array of actual angle values.
        velocity (np.ndarray): Array of actual velocity values.
        simulated_angle (np.ndarray): Array of simulated angle values.
        simulated_velocity (np.ndarray): Array of simulated velocity values.
        color (str): Color for the simulated data plot.
        alpha (float): Alpha value for the simulated data plot.
    """
    ax.plot(angle, velocity, label="Actual", color="black", linestyle="dashed", linewidth=2)
    ax.plot(simulated_angle, simulated_velocity, alpha=alpha, color=color, label="Simulated")
    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_title("Phase Portrait")
    ax.grid(color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    ax.legend()

def plot_simulation_errors(timespan: np.ndarray, angle: np.ndarray, velocity: np.ndarray, batched_states_trajectories: np.ndarray, predicted_terminal_points: np.ndarray, interval_terminal_states: np.ndarray, HORIZON: int, save_path: str = None, show: bool = False, title: str = "Simulation Errors", iteration: int = None) -> np.ndarray:
    """
    Plot simulation errors for the pendulum system and return the frame as an image.

    Args:
        timespan (np.ndarray): Array of time points.
        angle (np.ndarray): Array of actual angle values.
        velocity (np.ndarray): Array of actual velocity values.
        batched_states_trajectories (np.ndarray): Array of simulated state trajectories.
        predicted_terminal_points (np.ndarray): Array of predicted terminal points.
        interval_terminal_states (np.ndarray): Array of actual terminal states at intervals.
        HORIZON (int): Number of time steps in each interval.
        save_path (str): Path to save the plot. If None, the plot is not saved.
        show (bool): Whether to display the plot.
        title (str): Title for the plot.
        iteration (int, optional): Current iteration number for animation frames.

    Returns:
        np.ndarray: Image array representing the current frame.
    """
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    plot_state(ax1, timespan, angle, batched_states_trajectories[:, 0], "Angle")
    ax1.plot(timespan[HORIZON + 1 :][::HORIZON], predicted_terminal_points[:-1, 0], "ob", label="Predicted")
    ax1.plot(timespan[HORIZON + 1 :][::HORIZON], interval_terminal_states[:, 0], "or", label="Actual")
    if iteration is not None:
        ax1.set_title(f"{title} (Iteration {iteration})")
    else:
        ax1.set_title(title)
    ax1.legend(loc='upper right')

    plot_state(ax2, timespan, velocity, batched_states_trajectories[:, 1], "Velocity")
    ax2.plot(timespan[HORIZON + 1 :][::HORIZON], predicted_terminal_points[:-1, 1], "ob", label="Predicted")
    ax2.plot(timespan[HORIZON + 1 :][::HORIZON], interval_terminal_states[:, 1], "or", label="Actual")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc='upper right')

    plot_phase_portrait(ax3, angle, velocity, batched_states_trajectories[:, 0], batched_states_trajectories[:, 1])
    ax3.plot(predicted_terminal_points[:-1, 0], predicted_terminal_points[:-1, 1], "ob", label="Predicted")
    ax3.plot(interval_terminal_states[:, 0], interval_terminal_states[:, 1], "or", label="Actual")
    ax3.legend(loc='upper right')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    if show:
        plt.show()
    
    # Convert plot to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return image

def create_animation_frame(timespan: np.ndarray, true_trajectory: np.ndarray, current_rollout: np.ndarray, iteration: int) -> np.ndarray:
    """
    Create a single frame for the animation of the learning process.

    Args:
        timespan (np.ndarray): Array of time points.
        true_trajectory (np.ndarray): Array of actual state values.
        current_rollout (np.ndarray): Array of current simulated state values.
        iteration (int): Current iteration number.

    Returns:
        np.ndarray: Image array representing the current frame.
    """
    fig = plt.figure(figsize=(12, 6))  # Reduced height from 10 to 5
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    
    plot_state(ax1, timespan, true_trajectory[:, 0], current_rollout[:, 0], "Angle", color="red")
    ax1.set_title(f"Iteration {iteration}")

    plot_state(ax2, timespan, true_trajectory[:, 1], current_rollout[:, 1], "Velocity", color="red")
    ax2.set_xlabel("Time (s)")

    plot_phase_portrait(ax3, true_trajectory[:, 0], true_trajectory[:, 1], current_rollout[:, 0], current_rollout[:, 1], color="red")
    ax3.set_title("Phase Portrait")

    plt.tight_layout()
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return image

def plot_full_simulation(timespan: np.ndarray, angle: np.ndarray, velocity: np.ndarray, old_rollout: np.ndarray, new_rollout: np.ndarray, save_path: str = "plots/learning_results.png", show: bool = True) -> None:
    """
    Plot full simulation results for the pendulum system.

    Args:
        timespan (np.ndarray): Array of time points.
        angle (np.ndarray): Array of actual angle values.
        velocity (np.ndarray): Array of actual velocity values.
        old_rollout (np.ndarray): Array of simulated states using the old model.
        new_rollout (np.ndarray): Array of simulated states using the new model.
        save_path (str): Path to save the plot.
        show (bool): Whether to display the plot.
    """
    fig = plt.figure(figsize=(12, 6))  # Reduced height from 10 to 5
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    
    plot_state(ax1, timespan, angle, old_rollout[:, 0], "Angle", color="blue", alpha=0.3)
    ax1.plot(timespan, new_rollout[:, 0], color="red", label="Optimized Model")
    
    plot_state(ax2, timespan, velocity, old_rollout[:, 1], "Velocity", color="blue", alpha=0.3)
    ax2.plot(timespan, new_rollout[:, 1], color="red", label="Optimized Model")
    ax2.set_xlabel("Time (s)")

    plot_phase_portrait(ax3, angle, velocity, old_rollout[:, 0], old_rollout[:, 1], color="blue", alpha=0.3)
    ax3.plot(new_rollout[:, 0], new_rollout[:, 1], color="red", label="Optimized Model")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)
