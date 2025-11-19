import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"


class GraphDynamicsVisualizer:
    def __init__(
        self,
        ips,
        jumps,
        initial_state,
        output_file="graph_dynamics.mp4",
        fps=30,
        duration=None,
        state_colors=None,
        title=None,
        ode_solution=None,
        fontsize_scale=1.0,
    ):
        """
        Visualize graph dynamics and save as MP4.

        Parameters:
        -----------
        ips : IPS object
            The interacting particle system with graph
        jumps : list of tuples
            List of (node_id, jump_time, (source_state, target_state))
        initial_state : dict
            Dictionary mapping node_id to initial state
        output_file : str
            Output MP4 filename
        fps : int
            Frames per second for the video
        duration : float or None
            Total duration in seconds (if None, uses max event time + 0.5)
        state_colors : dict or None
            Mapping from state to color (e.g., {0: 'blue', 1: 'red', 2: 'green'})
        title : str or None
            Title for the graph subplot
        ode_solution : numpy array or None
            Pre-computed ODE solution with shape (num_states, num_frames)
            Will be plotted as dashed lines alongside simulation occupancy
        fontsize_scale : float
            Multiplier for all font sizes (default 1.0). Use 1.5 for 50% larger text, etc.
        """
        self.graph = ips.graph
        self.state_space = ips.state_space
        self.jumps = sorted(jumps, key=lambda x: x[1])  # Sort by time (second element)
        self.output_file = output_file
        self.fps = fps
        self.title = title
        self.ode_solution = ode_solution
        self.fontsize_scale = fontsize_scale

        # Store initial states
        self.initial_states = dict(initial_state)

        # Determine duration
        if duration is None:
            self.duration = self.jumps[-1][1] + 0.5 if self.jumps else 10
        else:
            self.duration = duration

        # Set up state colors
        if state_colors is None:
            try:
                from sparseips.visualization import get_color

                self.state_colors = {
                    state: get_color(i) for i, state in enumerate(ips.state_space)
                }
            except ImportError:
                # Fallback if get_color not available
                colors = [
                    "#3b82f6",
                    "#ef4444",
                    "#10b981",
                    "#f59e0b",
                    "#8b5cf6",
                    "#ec4899",
                ]
                self.state_colors = {
                    state: colors[i % len(colors)]
                    for i, state in enumerate(ips.state_space)
                }
        else:
            self.state_colors = state_colors

        # Compute layout once
        self.pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)

    def get_state_at_time(self, t):
        """Get the state of all nodes at time t."""
        states = dict(self.initial_states)  # Start from actual initial states
        recent = {}

        for node_id, event_time, (src_state, tgt_state) in self.jumps:
            if event_time <= t:
                states[node_id] = tgt_state
                # Track recent changes (within last 0.3 seconds)
                if t - event_time < 0.3:
                    recent[node_id] = (event_time, src_state, tgt_state)
            else:
                break

        return states, recent

    def create_animation(self):
        """Create and save the animation."""
        fig = plt.figure(figsize=(16, 9), facecolor="white")

        # Create subplots: main graph on left, time series on right
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)
        ax_graph = fig.add_subplot(gs[0])
        ax_timeseries = fig.add_subplot(gs[1])

        ax_graph.set_facecolor("white")
        ax_graph.axis("off")
        ax_graph.set_title(
            self.title,
            fontsize=int(16 * self.fontsize_scale),
            family="Times New Roman",
            pad=20,
        )

        # Calculate total frames
        total_frames = int(self.duration * self.fps)

        # Draw edges once (they don't change)
        nx.draw_networkx_edges(
            self.graph,
            self.pos,
            ax=ax_graph,
            edge_color="#666666",
            width=0.8,
            alpha=0.4,
        )

        # Create node collection that we'll update
        node_artists = {}
        ring_artists = {}

        for node in self.graph.nodes():
            # Main node circle - use actual initial state
            initial_state = self.initial_states.get(node, 0)
            circle = Circle(
                self.pos[node],
                radius=0.015,
                facecolor=self.state_colors.get(initial_state, "gray"),
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
            ax_graph.add_patch(circle)
            node_artists[node] = circle

            # Animation ring (initially invisible)
            ring = Circle(
                self.pos[node],
                radius=0.015,
                facecolor="none",
                edgecolor="black",
                linewidth=1.5,
                alpha=0,
                zorder=2,
            )
            ax_graph.add_patch(ring)
            ring_artists[node] = ring

        # Time text
        time_text = ax_graph.text(
            0.02,
            0.98,
            "",
            transform=ax_graph.transAxes,
            fontsize=int(14 * self.fontsize_scale),
            color="black",
            family="Times New Roman",
            verticalalignment="top",
        )

        # Legend for states
        legend_elements = []
        for state in self.state_space:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.state_colors[state],
                    markersize=10,
                    label=f"State {state}",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )
        legend = ax_graph.legend(
            handles=legend_elements,
            loc="upper right",
            framealpha=0.9,
            facecolor="white",
            edgecolor="black",
            prop={"family": "Times New Roman", "size": int(11 * self.fontsize_scale)},
        )

        # Get actual bounds of the layout
        x_vals = [pos[0] for pos in self.pos.values()]
        y_vals = [pos[1] for pos in self.pos.values()]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # Add padding
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax_graph.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax_graph.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
        ax_graph.set_aspect("equal")

        # Setup time series plot
        ax_timeseries.set_facecolor("white")
        ax_timeseries.set_xlim(0, self.duration)
        ax_timeseries.set_ylim(0, 1)
        ax_timeseries.set_xlabel(
            "time", fontsize=int(12 * self.fontsize_scale), family="Times New Roman"
        )
        ax_timeseries.set_ylabel(
            "state occupancy",
            fontsize=int(12 * self.fontsize_scale),
            family="Times New Roman",
        )
        ax_timeseries.grid(True, alpha=0.3)
        ax_timeseries.tick_params(labelsize=int(10 * self.fontsize_scale))

        # Create line objects for each state
        time_history = []
        occupancy_history = {state: [] for state in self.state_space}
        line_objects = {}
        ode_line_objects = {}

        for state in self.state_space:
            # Simulation line (solid)
            (line,) = ax_timeseries.plot(
                [], [], color=self.state_colors[state], linewidth=2, label=f"{state}"
            )
            line_objects[state] = line

            # ODE line (dashed) if provided
            if self.ode_solution is not None:
                (ode_line,) = ax_timeseries.plot(
                    [],
                    [],
                    color=self.state_colors[state],
                    linewidth=3,
                    linestyle="--",
                    alpha=0.7,
                )
                ode_line_objects[state] = ode_line

        # Update legend to indicate simulation vs ODE
        if self.ode_solution is not None:
            legend_handles = [
                plt.Line2D([0], [0], color="black", linewidth=2, label="Monte Carlo"),
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    label="Local-field prediction",
                ),
            ]
            ax_timeseries.legend(
                handles=legend_handles,
                loc="upper left",
                framealpha=0.9,
                facecolor="white",
                edgecolor="black",
                prop={
                    "family": "Times New Roman",
                    "size": int(10 * self.fontsize_scale),
                },
            )
        # else:
        #     ax_timeseries.legend(loc='upper left', framealpha=0.9,
        #                          facecolor='white', edgecolor='black',
        #                          prop={'family': 'Times New Roman', 'size': int(10 * self.fontsize_scale)})

        def update(frame):
            """Update function for animation."""
            t = frame / self.fps

            # Get states at this time
            states, recent = self.get_state_at_time(t)

            # Update nodes
            for node in self.graph.nodes():
                state = states[node]
                color = self.state_colors.get(state, "gray")
                node_artists[node].set_facecolor(color)

                # Update animation ring for recent changes
                if node in recent:
                    event_time, _, _ = recent[node]
                    age = t - event_time
                    alpha = max(0, 1 - age / 0.3)
                    radius = 0.015 + age * 0.08
                    ring_artists[node].set_radius(radius)
                    ring_artists[node].set_alpha(alpha)
                    ring_artists[node].set_edgecolor(color)
                else:
                    ring_artists[node].set_alpha(0)

            # Update time text
            time_text.set_text(f"t = {t:.2f}s")

            # Update time series
            time_history.append(t)
            total_nodes = len(self.graph.nodes())

            # Calculate occupancy for each state
            for state in self.state_space:
                count = sum(1 for node_state in states.values() if node_state == state)
                occupancy = count / total_nodes if total_nodes > 0 else 0
                occupancy_history[state].append(occupancy)

            # Update line data
            for state in self.state_space:
                line_objects[state].set_data(time_history, occupancy_history[state])

            # Update ODE lines if provided
            artists_to_return = (
                list(node_artists.values())
                + list(ring_artists.values())
                + [time_text]
                + list(line_objects.values())
            )

            if self.ode_solution is not None:
                for i, state in enumerate(self.state_space):
                    # Get ODE data up to current frame
                    ode_line_objects[state].set_data(
                        time_history, self.ode_solution[i, : len(time_history)]
                    )
                artists_to_return += list(ode_line_objects.values())

            return artists_to_return

        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=total_frames, interval=1000 / self.fps, blit=True
        )

        # Save as MP4
        print(f"Generating {total_frames} frames at {self.fps} FPS...")

        # Try ffmpeg first, fall back to pillow if not available
        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(
                fps=self.fps,
                bitrate=2000,
                extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
            )
            anim.save(self.output_file, writer=writer, dpi=100)
            print(f"Video saved to {self.output_file}")
        except (KeyError, RuntimeError) as e:
            print("ffmpeg not found. Trying pillow writer (will save as GIF)...")
            try:
                # Change extension to .gif
                gif_file = self.output_file.replace(".mp4", ".gif")
                Writer = animation.PillowWriter(fps=self.fps)
                anim.save(gif_file, writer=Writer, dpi=100)
                print(f"Animation saved as GIF to {gif_file}")
                print("Note: Install ffmpeg for MP4 output with better compression")
            except Exception as e2:
                print(f"Error: Could not save animation. {e2}")
                print("Please install ffmpeg or pillow: pip install pillow")

        plt.close()
