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


def plot_particle_trajectories(
        jumps: List[Tuple[int, float, Tuple[Any, Any]]],
        graph: nx.Graph,
        state_space: List[Any],
        initial_conditions: Dict[int, Any],
        max_time: float,
        plot_type: str = "both",
        figsize: Tuple[int, int] = (14, 10),
        state_colors: Dict = None,
        node_positions: Dict = None,
        max_nodes_to_plot: int = 20
) -> None:
    """
    Visualize the trajectories of particles from jump process simulation.

    Parameters:
    -----------
    jumps : List[Tuple[int, float, Tuple[Any, Any]]]
        Output from simulate_jump_process function
    graph : nx.Graph
        The graph structure used in the simulation
    state_space : List[Any]
        List of possible states a particle can take
    initial_conditions : Dict[int, Any]
        Dictionary mapping node indices to their initial states
    max_time : float
        Maximum simulation time
    plot_type : str, optional
        Type of plot to generate: "trajectories", "snapshots", or "both"
    figsize : Tuple[int, int], optional
        Figure size
    state_colors : Dict, optional
        Dictionary mapping states to colors
    node_positions : Dict, optional
        Dictionary mapping node indices to (x,y) positions
    max_nodes_to_plot : int, optional
        Maximum number of nodes to show in trajectory plot (for large graphs)
    """
    # Create new figure
    plt.figure(figsize=figsize)

    # If no state colors are provided, generate them
    if state_colors is None:
        # Use more distinctive colors
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                  '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb']
        state_colors = {state: colors[i % len(colors)] for i, state in enumerate(state_space)}

    # If no node positions are provided, generate them
    if node_positions is None:
        node_positions = nx.spring_layout(graph, seed=42)

    # Organize jumps by vertex
    vertex_jumps = defaultdict(list)
    for vertex, time, transition in jumps:
        vertex_jumps[vertex].append((time, transition))

    # Create time series of states for each vertex
    vertex_states = {}
    for vertex in graph.nodes():
        # Start with initial state
        state_history = [(0, initial_conditions[vertex])]

        # Add all transitions
        for time, (_, target_state) in vertex_jumps[vertex]:
            state_history.append((time, target_state))

        # End with final state at max_time
        if state_history[-1][0] < max_time:
            state_history.append((max_time, state_history[-1][1]))

        vertex_states[vertex] = state_history

    # Plot based on requested type
    if plot_type in ["trajectories", "both"]:
        if plot_type == "both":
            plt.subplot(2, 1, 1)

        # For large graphs, choose a subset of nodes to plot
        nodes_to_plot = list(graph.nodes())
        if len(nodes_to_plot) > max_nodes_to_plot:
            nodes_to_plot = sorted(nodes_to_plot)[:max_nodes_to_plot]
            plt.title(f'Particle State Trajectories (showing {max_nodes_to_plot} of {len(graph.nodes())} nodes)')
        else:
            plt.title('Particle State Trajectories')

        # Plot trajectories as piecewise constant functions
        for vertex in nodes_to_plot:
            state_history = vertex_states[vertex]

            # Create piecewise constant representation
            pc_times = []
            pc_states = []
            pc_colors = []

            for i in range(len(state_history) - 1):
                t_current, state = state_history[i]
                t_next, _ = state_history[i + 1]

                # Add two points for the current state (forming horizontal line)
                pc_times.extend([t_current, t_next])
                pc_states.extend([state_space.index(state), state_space.index(state)])
                pc_colors.extend([state_colors[state], state_colors[state]])

            # Plot step function
            plt.step(pc_times, pc_states, where='post', label=f'Vertex {vertex}',
                     color=f'C{vertex % 10}', alpha=0.7)

            # Add markers at transition points
            transition_times = [t for t, _ in state_history]
            transition_states = [state_space.index(s) for _, s in state_history]
            plt.scatter(transition_times, transition_states,
                        color=[state_colors[s] for _, s in state_history],
                        marker='o', s=50, zorder=3)

        plt.yticks(range(len(state_space)), state_space)
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend only if there aren't too many nodes
        if len(nodes_to_plot) <= 10:
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    if plot_type in ["snapshots", "both"]:
        subplot_pos = 2 if plot_type == "both" else 1

        # Create snapshots of the graph at different times
        num_snapshots = min(5, len(jumps) + 1)
        snapshot_times = np.linspace(0, max_time, num_snapshots)

        for i, t in enumerate(snapshot_times):
            if plot_type == "both":
                plt.subplot(2, num_snapshots, subplot_pos * num_snapshots - num_snapshots + i + 1)
            else:
                plt.subplot(1, num_snapshots, i + 1)

            # Get states at time t
            current_states = {}
            for vertex, state_history in vertex_states.items():
                # Find the state at time t
                idx = 0
                while idx < len(state_history) - 1 and state_history[idx + 1][0] <= t:
                    idx += 1
                current_states[vertex] = state_history[idx][1]

            # Draw graph with node colors based on states
            node_colors = [state_colors[current_states[n]] for n in graph.nodes()]

            # For large graphs, adjust node size
            node_size = 500 if len(graph.nodes()) <= 25 else max(50, 2000 // len(graph.nodes()))
            font_size = 10 if len(graph.nodes()) <= 25 else max(4, 200 // len(graph.nodes()))

            nx.draw_networkx(
                graph,
                pos=node_positions,
                node_color=node_colors,
                with_labels=len(graph.nodes()) <= 100,  # Only show labels for smaller graphs
                node_size=node_size,
                font_size=font_size,
                font_weight='bold',
                font_color='black',
                edge_color='gray',
                width=0.5,
                alpha=0.8
            )
            plt.title(f'Time: {t:.2f}')
            plt.axis('off')

    # Add a color legend for the states
    if plot_type == "snapshots" or (plot_type == "both" and len(nodes_to_plot) > 10):
        # Create a small custom legend for the states
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=state_colors[s], markersize=10, label=f'State {s}')
                   for s in state_space]
        if plot_type == "both":
            plt.figlegend(handles=handles, loc='lower right', bbox_to_anchor=(0.95, 0.05))
        else:
            plt.figlegend(handles=handles, loc='lower right', bbox_to_anchor=(0.95, 0.05))

    plt.tight_layout()
    plt.show()


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
        figsize=(10, 6)
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
        colors = cm.tab10(np.linspace(0, 1, len(ips.state_space)))

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
        all_states = get_particle_states_at_times(jumps, initial_conditions_list[run_idx], time_points)

        # Process each time point
        for t_idx, states_at_time in enumerate(all_states):
            # Calculate empirical probabilities
            counter = Counter(states_at_time.values())
            for s_idx, state in enumerate(ips.state_space):
                all_probabilities[run_idx, s_idx, t_idx] = counter.get(state, 0) / num_nodes

    # Calculate mean and standard deviation across runs
    mean_probabilities = np.mean(all_probabilities, axis=0)
    std_probabilities = np.std(all_probabilities, axis=0) / np.sqrt(num_runs)

    # Plot the mean empirical probabilities as step functions with error bands
    for s_idx, state in enumerate(ips.state_space):
        line = ax.step(time_points, mean_probabilities[s_idx], where='post',
                       color=colors[s_idx], label=labels[s_idx], alpha=0.8)

        # Add error bands (2 standard deviations)
        lower_bound = np.maximum(0, mean_probabilities[s_idx] - 2 * std_probabilities[s_idx])
        upper_bound = np.minimum(1, mean_probabilities[s_idx] + 2 * std_probabilities[s_idx])

        ax.fill_between(time_points, lower_bound, upper_bound,
                        step='post', alpha=0.2, color=colors[s_idx])

    # Add legend, labels, and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Empirical Probability')
    ax.set_title(f'Average state-occupancy for {ips.name}' if title is None else title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)

    # Set x-axis limits
    ax.set_xlim(0, max_time)

    # add grid
    ax.grid(True, alpha=0.3)

    return fig, ax


def animate_particle_process(
        jumps: List[Tuple[int, float, Tuple[Any, Any]]],
        graph: nx.Graph,
        state_space: List[Any],
        initial_conditions: Dict[int, Any],
        max_time: float,
        state_colors: Dict = None,
        node_positions: Dict = None,
        show_animation: bool = True,
        fps: int = 10,
        num_frames: int = 100,
        save_path: str = None,
        figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Create an animation of particles evolving on a graph.

    Parameters:
    -----------
    jumps : List[Tuple[int, float, Tuple[Any, Any]]]
        Output from simulate_jump_process function
    graph : nx.Graph
        The graph structure used in the simulation
    state_space : List[Any]
        List of possible states a particle can take
    initial_conditions : Dict[int, Any]
        Dictionary mapping node indices to their initial states
    max_time : float
        Maximum simulation time
    state_colors : Dict, optional
        Dictionary mapping states to colors
    node_positions : Dict, optional
        Dictionary mapping node indices to (x,y) positions
    show_animation : bool, optional
        If True, display the animation
    fps : int, optional
        Frames per second in the animation
    num_frames : int, optional
        Number of frames in the animation
    save_path : str, optional
        If provided, save the animation to this path
    figsize : Tuple[int, int], optional
        Figure size
    """
    # If no state colors are provided, generate them
    if state_colors is None:
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
                  '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb']
        state_colors = {state: colors[i % len(colors)] for i, state in enumerate(state_space)}

    # If no node positions are provided, generate them
    if node_positions is None:
        node_positions = nx.spring_layout(graph, seed=42)

    # Organize jumps by vertex
    vertex_jumps = defaultdict(list)
    for vertex, time, transition in jumps:
        vertex_jumps[vertex].append((time, transition))

    # Create time series of states for each vertex
    vertex_states = {}
    for vertex in graph.nodes():
        # Start with initial state
        state_history = [(0, initial_conditions[vertex])]

        # Add all transitions
        for time, (_, target_state) in vertex_jumps[vertex]:
            state_history.append((time, target_state))

        # End with final state at max_time
        if state_history[-1][0] < max_time:
            state_history.append((max_time, state_history[-1][1]))

        vertex_states[vertex] = state_history

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    plt.close()  # We'll display the animation separately

    # Function to get states at a given time
    def get_states_at_time(t):
        current_states = {}
        for vertex, state_history in vertex_states.items():
            # Find the state at time t
            idx = 0
            while idx < len(state_history) - 1 and state_history[idx + 1][0] <= t:
                idx += 1
            current_states[vertex] = state_history[idx][1]
        return current_states

    # For large graphs, adjust node size
    node_size = 500 if len(graph.nodes()) <= 25 else max(50, 2000 // len(graph.nodes()))
    font_size = 10 if len(graph.nodes()) <= 25 else max(4, 200 // len(graph.nodes()))

    # Animation initialization function
    def init():
        ax.clear()
        ax.set_title('Time: 0.00')
        ax.axis('off')
        return []

    # Animation update function
    def update(frame):
        ax.clear()
        t = frame * max_time / (num_frames - 1)
        current_states = get_states_at_time(t)

        # Draw graph with node colors based on states
        node_colors = [state_colors[current_states[n]] for n in graph.nodes()]

        nx.draw_networkx(
            graph,
            pos=node_positions,
            node_color=node_colors,
            with_labels=len(graph.nodes()) <= 100,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            font_color='black',
            edge_color='gray',
            width=0.5,
            alpha=0.8,
            ax=ax
        )

        # Create a small custom legend for the states
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=state_colors[s], markersize=10, label=f'State {s}')
                   for s in state_space]
        ax.legend(handles=handles, loc='upper right', framealpha=0.7)

        ax.set_title(f'Time: {t:.2f}')
        ax.axis('off')
        return []

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  init_func=init, blit=True, interval=1000 / fps)

    # Save animation if requested
    if save_path:
        Writer = animation.writers['ffmpeg'] if 'ffmpeg' in animation.writers.list() else animation.writers['pillow']
        writer = Writer(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    # Display animation
    if show_animation:
        plt.show()

    return ani