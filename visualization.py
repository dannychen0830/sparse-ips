from sparseips.jump_ips_sim import get_particle_states_at_times
from sparseips.ips_class import ParticleSystem
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import matplotlib.animation as animation
from matplotlib.pyplot import cm
from collections import Counter


def get_color(idx, num_colors=10):
    return (
        cm.tab10(np.linspace(0, 1, num_colors))
        if idx is None
        else cm.tab10(np.linspace(0, 1, num_colors))[idx]
    )


def plot_ensemble_state_probabilities(
        ips: ParticleSystem,
        jumps_list: List[List[Tuple[int, float, Tuple[Any, Any]]]],
        initial_conditions_list: List[Dict[int, Any]],
        max_time: float,
        num_samples: int = 100,
        ax=None,
        colors=None,
        labels=None,
        title: str = None,
        figsize=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the average empirical probability of states across multiple simulation runs,
    with error bars representing the standard deviation across runs.

    Uses an optimized method to calculate states at multiple time points efficiently.

    Parameters:
    -----------
    jumps_list : List[List[Tuple[int, float, Tuple[Any, Any]]]]
        List of outputs from multiple calls to simulate_jump_process
    initial_conditions : Dict[int, Any]
        Dictionary mapping node indices to their initial states
    state_space : List[Any]
        List of possible states a particle can take
    max_time : float
        Maximum simulation time from the simulation
    num_samples : int, optional
        Number of time points to sample for the plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    colors : List[str], optional
        Colors for each state. If None, a default color cycle is used.
    labels : List[str], optional
        Labels for each state. If None, str(state) is used.
    title : str, optional
        Title for the plot
    figsize : Tuple[float, float], optional
        Figure size if ax is None

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set default colors if not provided
    if colors is None:
        colors = get_color(None, max(len(ips.state_space), 10))

    # Set default labels if not provided
    if labels is None:
        labels = [str(state) for state in ips.state_space]

    # Generate time points to evaluate
    time_points = np.linspace(0, max_time, num_samples)

    # Get number of nodes and runs
    num_nodes = len(initial_conditions_list[0])
    num_runs = len(jumps_list)

    # Initialize array to store probabilities for all runs
    all_probabilities = np.zeros((num_runs, len(ips.state_space), num_samples))

    # For each run, calculate empirical probabilities at all time points efficiently
    for run_idx, jumps in enumerate(jumps_list):
        # Get system state at all time points in one pass
        all_states = get_particle_states_at_times(
            jumps, initial_conditions_list[run_idx], time_points
        )

        # Process each time point
        for t_idx, states_at_time in enumerate(all_states):
            # Calculate empirical probabilities
            counter = Counter(states_at_time.values())
            for s_idx, state in enumerate(ips.state_space):
                all_probabilities[run_idx, s_idx, t_idx] = (
                        counter.get(state, 0) / num_nodes
                )

    # Calculate mean and standard deviation across runs
    mean_probabilities = np.mean(all_probabilities, axis=0)
    std_probabilities = np.std(all_probabilities, axis=0) / np.sqrt(num_runs)

    # Plot the mean empirical probabilities as step functions with error bands
    for s_idx, state in enumerate(ips.state_space):
        line = ax.plot(
            time_points,
            mean_probabilities[s_idx],
            color=colors[s_idx],
            label=labels[s_idx],
            alpha=0.8,
        )

        # Add error bands (2 standard deviations)
        lower_bound = np.maximum(
            0, mean_probabilities[s_idx] - 2 * std_probabilities[s_idx]
        )
        upper_bound = np.minimum(
            1, mean_probabilities[s_idx] + 2 * std_probabilities[s_idx]
        )

        ax.fill_between(
            time_points, lower_bound, upper_bound, alpha=0.2, color=colors[s_idx]
        )

    # Add legend, labels, and title
    ax.set_xlabel("time")
    ax.set_ylabel(None)
    ax.set_title(f"Average state-occupancy for {ips.name}" if title is None else title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)

    # Set x-axis limits
    ax.set_xlim(0, max_time)

    # add grid
    ax.grid(True, alpha=0.3)

    return fig, ax
