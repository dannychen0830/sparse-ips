import numpy as np
from sparseips.ips_class import *
from joblib import Parallel, delayed # use this to parallelize also in jupyter notebooks


def simulate_jump_process(
    ips: ParticleSystem,
    initial_conditions: dict[int,any],
    max_time: float,
    seed: int = None,
    verbose: bool = False,
    edge_initial_conditions: dict[tuple[int,int],any] = None,
    return_edge_jumps: bool = False,
) -> list[tuple[int, float, tuple[any, any]]]:
    """
    Simulate interacting particles on a graph as a continuous-time Markov chain.

    Parameters:
    -----------
    state_space : list[Any]
        List of possible states a particle can take
    graph : nx.Graph
        Graph that specifies the interaction structure
    rate : Callable
        Function that takes the source state, target state, and:
        - If it has 3 parameters: only neighbor states
        - If it has 4 parameters: neighbor states and global state
        The signature is automatically detected
    initial_conditions : dict[int,any
        Dictionary mapping node indices to their initial states
    max_time : float
        Maximum simulation time
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    list[tuple[int, float, tuple[any, any]]
        List of tuples (vertex that jumped, jump time, transition that occurred)
        where transition is a tuple (source_state, target_state)
    """
    if isinstance(ips, MeanFieldParticleSystem):
        return simulate_mean_field_jump_process(
            ips, initial_conditions, max_time, seed, verbose
        )

    if seed is not None:
        np.random.seed(seed)
    if verbose:
        start_time = np.datetime64("now")

    # Initialize current state and simulation time
    current_vertex_state = initial_conditions.copy()
    current_edge_state = edge_initial_conditions.copy() if edge_initial_conditions is not None else None
    current_time = 0.0

    # Initialize global empirical measure if needed
    meas = ips.compute_global_empirical_measure(current_vertex_state) if ips.global_interaction else None

    # Initialize results list
    jumps = []
    edge_jumps = []

    while current_time < max_time:
        # Calculate rates for all possible transitions
        possible_transitions = []
        rates = []
 
        for node in ips.graph.nodes():
            current_node_state = current_vertex_state[node]

            # Consider all possible target states for this node
            for target_state in ips.state_space:
                if target_state != current_node_state:
                    transition_rate = ips.sim_rate(
                        node=node,
                        source_state=current_node_state,
                        target_state=target_state,
                        current_config=current_vertex_state,
                        t=current_time,
                        meas=meas,
                        current_edge_state=current_edge_state,
                    )

                    if transition_rate > 0:
                        possible_transitions.append(
                            (node, current_node_state, target_state)
                        )
                        rates.append(transition_rate)

        # if available, add the edge weight dynamics
        if ips.edge_rate is not None:
            for edge in ips.graph.edges():
                # sort edge 
                edge = tuple(sorted(edge))
                
                for target_state in ips.edge_state_space:
                    transition_rate = ips.edge_sim_rate(
                        edge,
                        current_edge_state[edge],
                        target_state,
                        current_vertex_state,
                        meas=meas,
                        t=current_time,
                    )

                    if transition_rate > 0:
                        possible_transitions.append(
                            (edge, current_edge_state[edge], target_state)
                        )
                        rates.append(transition_rate)

    
        # If no transitions are possible, we're done
        if not rates:
            break

        # Calculate total rate
        total_rate = np.sum(rates)

        # Sample time to next event (exponential distribution)
        if total_rate > 0:
            time_to_next_event = np.random.exponential(1.0 / total_rate)
        else:
            break

        # Update time
        current_time += time_to_next_event

        if current_time > max_time:
            # If we exceeded max_time, we stop
            break

        # Sample which transition occurs
        transition_index = np.random.choice(
            len(possible_transitions), p=np.array(rates) / np.sum(rates)
        )

        entity, source_state, target_state = possible_transitions[transition_index]

        # Update state
        if isinstance(entity, tuple):
            edge = tuple(sorted(entity))
            current_edge_state[edge] = target_state
            edge_jumps.append((edge, current_time, (source_state, target_state)))
        else:
            node = entity
            affected_nodes = [node] + list(ips.graph.neighbors(node))

            if meas is not None:
                for v in affected_nodes:
                    old_nb = ips.get_neighborhood(v, current_vertex_state)
                    meas[old_nb] = meas.get(old_nb, 0.0) - 1.0 / ips.num_particles
                    if meas[old_nb] <= 1e-12:
                        del meas[old_nb]

            current_vertex_state[node] = target_state
            jumps.append((node, current_time, (source_state, target_state)))

            if meas is not None:
                for v in affected_nodes:
                    new_nb = ips.get_neighborhood(v, current_vertex_state)
                    meas[new_nb] = meas.get(new_nb, 0.0) + 1.0 / ips.num_particles

    if verbose:
        print(f'completed simulation in {np.datetime64("now") - start_time} seconds.')

    if return_edge_jumps:
        return jumps, edge_jumps
    return jumps


def simulate_sir_mc_parallel(ips, initial_conditions, num_sims=15, max_time=5, seed=10, n_jobs=-1):
    """
    Parallelized version using joblib (works better in notebooks).
    n_jobs=-1 uses all CPUs.
    """
    index_to_state = ips.get_state_to_index_map()
    index_to_state = {v: k for k, v in index_to_state.items()}

    def single_sim(t):
        sim_init_cond = {
            i: index_to_state[np.random.choice(len(ips.state_space),
                                                   p=[p for p in initial_conditions.values()])]
            for i in range(ips.num_particles)
        }

        jumps = simulate_jump_process(
            ips=ips,
            initial_conditions=sim_init_cond,
            max_time=max_time,
            seed=seed + t,
            verbose=False
        )

        return sim_init_cond, jumps

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(single_sim)(t) for t in range(num_sims)
    )

    # Unpack results
    sim_init_cond_list = [result[0] for result in results]
    jumps_list = [result[1] for result in results]

    return sim_init_cond_list, jumps_list


def get_particle_states_at_times(
    jumps: list[tuple[int, float, tuple[any, any]]],
    initial_conditions: dict[int,any],
    timestamps: list[float],
) -> list[dict[int,any]]:
    """
    Get the state of each particle at multiple specific time points efficiently.

    Parameters:
    -----------
    jumps : list[tuple[int, float, tuple[any, any]]
        Output from simulate_jump_process function
    initial_conditions : dict[int,any
        Dictionary mapping particle indices to their initial states
    timestamps : list[float]
        List of time points to query

    Returns:
    --------
    list[dict[int,any]
        List of dictionaries mapping particle indices to their states at each time point,
        in the same order as the input timestamps
    """
    # Sort timestamps in ascending order and keep track of original indices
    sorted_times_with_indices = sorted(enumerate(timestamps), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sorted_times_with_indices]
    sorted_times = [t for _, t in sorted_times_with_indices]

    # Initialize results for all timestamps with the initial conditions
    results = [{k: v for k, v in initial_conditions.items()} for _ in timestamps]

    # Sort jumps by time (they should already be sorted, but just to be safe)
    sorted_jumps = sorted(jumps, key=lambda x: x[1])

    time_index = 0
    current_states = {k: v for k, v in initial_conditions.items()}

    # Process jumps in time order
    for particle_idx, jump_time, (_, target_state) in sorted_jumps:
        # Update results for all timestamps between the last processed jump and this one
        while time_index < len(sorted_times) and sorted_times[time_index] < jump_time:
            results[sorted_indices[time_index]] = {
                k: v for k, v in current_states.items()
            }
            time_index += 1

        # Update current state
        current_states[particle_idx] = target_state

        # If we've processed all timestamps, we can exit early
        if time_index >= len(sorted_times):
            break

    # Handle any remaining timestamps after the last jump
    while time_index < len(sorted_times):
        results[sorted_indices[time_index]] = {k: v for k, v in current_states.items()}
        time_index += 1

    return results


def simulate_mean_field_jump_process(
    mfps: MeanFieldParticleSystem,
    initial_conditions: dict[int,any],
    max_time: float,
    seed: int = None,
    verbose: bool = False,
    tau_leap: bool = False,
    tau: float = 0.01,
    epsilon: float = 0.03,
) -> list[tuple[int, float, tuple[any, any]]]:
    """
    Simulate a mean-field particle system as a continuous-time Markov chain.

    Parameters:
    -----------
    mfps : MeanFieldParticleSystem
        The mean-field particle system to simulate
    initial_conditions : dict[int,any
        Dictionary mapping particle indices to their initial states
    max_time : float
        Maximum simulation time
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Whether to print timing information
    tau_leap : bool, optional
        Whether to use tau-leaping algorithm for faster simulation
    tau : float, optional
        Tau-leaping time step (only used if tau_leap=True)
    epsilon : float, optional
        Tau-leaping error control parameter (only used if tau_leap=True)

    Returns:
    --------
    list[tuple[int, float, tuple[any, any]]
        List of tuples (particle_id, jump_time, transition)
        where transition is a tuple (source_state, target_state)
    """
    if tau_leap:
        return simulate_mean_field_tau_leap(
            mfps, initial_conditions, max_time, seed, verbose, tau, epsilon
        )
    else:
        return simulate_mean_field_exact(
            mfps, initial_conditions, max_time, seed, verbose
        )


def simulate_mean_field_exact(
    mfps: MeanFieldParticleSystem,
    initial_conditions: dict[int,any],
    max_time: float,
    seed: int = None,
    verbose: bool = False,
) -> list[tuple[int, float, tuple[any, any]]]:
    """Exact simulation (original algorithm)."""
    if seed is not None:
        np.random.seed(seed)
    if verbose:
        start_time = np.datetime64("now")

    # Initialize current state and simulation time
    current_state = initial_conditions.copy()
    current_time = 0.0

    # Initialize empirical measure and maintain it throughout
    empirical_measure = mfps.compute_empirical_measure(current_state)

    # Create and maintain particle lists by state for efficient sampling
    particles_by_state = {}
    for particle_id, state in current_state.items():
        if state not in particles_by_state:
            particles_by_state[state] = []
        particles_by_state[state].append(particle_id)

    # Initialize results list
    jumps = []

    while current_time < max_time:
        # Calculate rates for all possible STATE transitions (not per particle)
        possible_transitions = []
        rates = []

        for source_state in mfps.state_space:
            # Only consider transitions from states that have particles
            if (
                source_state not in empirical_measure
                or empirical_measure[source_state] == 0
            ):
                continue

            for target_state in mfps.state_space:
                if target_state != source_state:
                    # Rate per particle in source state
                    per_particle_rate = mfps.sim_rate(
                        source_state, target_state, empirical_measure
                    )

                    if per_particle_rate > 0:
                        # Total rate = per_particle_rate × number_of_particles_in_source_state
                        num_particles_in_source = int(
                            empirical_measure[source_state] * mfps.num_particles
                        )
                        total_transition_rate = (
                            per_particle_rate * num_particles_in_source
                        )

                        possible_transitions.append((source_state, target_state))
                        rates.append(total_transition_rate)

        # If no transitions are possible, we're done
        if not rates:
            break

        # Calculate total rate
        total_rate = sum(rates)

        # Sample time to next event (exponential distribution)
        if total_rate > 0:
            time_to_next_event = np.random.exponential(1.0 / total_rate)
        else:
            break

        # Update time
        current_time += time_to_next_event

        if current_time > max_time:
            # If we exceeded max_time, we stop
            break

        # Sample which STATE transition occurs
        transition_index = np.random.choice(
            len(possible_transitions), p=np.array(rates) / total_rate
        )
        source_state, target_state = possible_transitions[transition_index]

        # Now randomly choose a particle in the source state
        available_particles = particles_by_state[source_state]
        particle_id = np.random.choice(available_particles)

        # Update state
        current_state[particle_id] = target_state

        # Update particle lists by state
        particles_by_state[source_state].remove(particle_id)
        if target_state not in particles_by_state:
            particles_by_state[target_state] = []
        particles_by_state[target_state].append(particle_id)

        # Remove empty state lists to keep dictionary clean
        if len(particles_by_state[source_state]) == 0:
            del particles_by_state[source_state]

        # EFFICIENTLY UPDATE EMPIRICAL MEASURE
        # Decrease count for source state, increase count for target state
        empirical_measure[source_state] = (
            empirical_measure.get(source_state, 0) - 1.0 / mfps.num_particles
        )
        empirical_measure[target_state] = (
            empirical_measure.get(target_state, 0) + 1.0 / mfps.num_particles
        )

        # Clean up zero entries to keep dictionary clean
        if empirical_measure[source_state] == 0:
            del empirical_measure[source_state]

        # Record jump
        jumps.append((particle_id, current_time, (source_state, target_state)))

    if verbose:
        print(f'completed simulation in {np.datetime64("now") - start_time} seconds.')

    return jumps


def simulate_mean_field_tau_leap(
    mfps: MeanFieldParticleSystem,
    initial_conditions: dict[int,any],
    max_time: float,
    seed: int = None,
    verbose: bool = False,
    tau: float = 0.01,
    epsilon: float = 0.03,
) -> list[tuple[int, float, tuple[any, any]]]:
    """
    Simulate using tau-leaping for faster approximate simulation.

    Parameters:
    -----------
    tau : float
        Fixed time step for tau-leaping
    epsilon : float
        Error control parameter for adaptive tau selection
    """
    if seed is not None:
        np.random.seed(seed)
    if verbose:
        start_time = np.datetime64("now")

    # Initialize current state and simulation time
    current_state = initial_conditions.copy()
    current_time = 0.0

    # Initialize empirical measure
    empirical_measure = mfps.compute_empirical_measure(current_state)

    # Track state counts for efficient updates
    state_counts = {}
    for state in mfps.state_space:
        state_counts[state] = sum(1 for s in current_state.values() if s == state)

    # Create particle lists by state for sampling
    particles_by_state = {}
    for particle_id, state in current_state.items():
        if state not in particles_by_state:
            particles_by_state[state] = []
        particles_by_state[state].append(particle_id)

    # Initialize results list
    jumps = []

    while current_time < max_time:
        # Calculate current rates for all transitions
        transition_rates = {}
        total_rate = 0.0

        for source_state in mfps.state_space:
            if state_counts.get(source_state, 0) == 0:
                continue

            for target_state in mfps.state_space:
                if target_state != source_state:
                    per_particle_rate = mfps.sim_rate(
                        source_state, target_state, empirical_measure
                    )

                    if per_particle_rate > 0:
                        total_transition_rate = (
                            per_particle_rate * state_counts[source_state]
                        )
                        transition_rates[(source_state, target_state)] = (
                            total_transition_rate
                        )
                        total_rate += total_transition_rate

        if total_rate == 0:
            break

        # Adaptive tau selection (optional - can use fixed tau for speed)
        if epsilon > 0:
            # Simple adaptive tau: ensure no state changes by more than epsilon*N in one step
            max_expected_change = (
                max(transition_rates.values()) * tau if transition_rates else 0
            )
            if max_expected_change > epsilon * mfps.num_particles:
                tau = min(
                    tau, epsilon * mfps.num_particles / max(transition_rates.values())
                )

        # Ensure we don't overshoot max_time
        actual_tau = min(tau, max_time - current_time)
        if actual_tau <= 0:
            break

        # Sample number of each type of transition in this time step
        transition_counts = {}
        for (source_state, target_state), rate in transition_rates.items():
            # Poisson sample with rate * tau
            expected_jumps = rate * actual_tau
            if expected_jumps > 0:
                num_jumps = np.random.poisson(expected_jumps)
                # Can't have more jumps than particles in source state
                num_jumps = min(num_jumps, state_counts.get(source_state, 0))
                if num_jumps > 0:
                    transition_counts[(source_state, target_state)] = num_jumps

        # Apply all transitions
        for (source_state, target_state), num_jumps in transition_counts.items():
            if num_jumps == 0:
                continue

            # Select particles to transition
            available_particles = particles_by_state.get(source_state, [])
            if len(available_particles) < num_jumps:
                num_jumps = len(available_particles)

            # Randomly select particles to transition
            transitioning_particles = np.random.choice(
                available_particles, size=num_jumps, replace=False
            )

            for particle_id in transitioning_particles:
                # Update state
                current_state[particle_id] = target_state

                # Update particle lists
                particles_by_state[source_state].remove(particle_id)
                if target_state not in particles_by_state:
                    particles_by_state[target_state] = []
                particles_by_state[target_state].append(particle_id)

                # Record jump (assign random time within the tau interval)
                jump_time = current_time + np.random.uniform(0, actual_tau)
                jumps.append((particle_id, jump_time, (source_state, target_state)))

            # Update state counts
            state_counts[source_state] -= num_jumps
            state_counts[target_state] = state_counts.get(target_state, 0) + num_jumps

        # Update empirical measure
        empirical_measure = {
            state: count / mfps.num_particles
            for state, count in state_counts.items()
            if count > 0
        }

        # Clean up empty particle lists
        particles_by_state = {
            state: particles
            for state, particles in particles_by_state.items()
            if particles
        }

        # Advance time
        current_time += actual_tau

    # Sort jumps by time (since tau-leaping can create out-of-order jumps)
    jumps.sort(key=lambda x: x[1])

    if verbose:
        print(
            f'completed tau-leaping simulation in {np.datetime64("now") - start_time} seconds.'
        )

    return jumps
