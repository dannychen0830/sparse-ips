import numpy as np
from sparseips.ips_class import ParticleSystem, MeanFieldParticleSystem
from sparseips.jump_ips_sim import simulate_mean_field_jump_process, get_particle_states_at_times

from jax.experimental import sparse
import jax
import jax.numpy as jnp
import diffrax
import lineax as lx
import optimistix as optx

import gc
from itertools import product
from collections import Counter


def one_coordinate_apart(tuple1: tuple, tuple2: tuple) -> bool:
    """
    Check if two tuples are one coordinate apart, i.e., they differ in exactly one coordinate.
    :param tuple1: first input tuple
    :param tuple2: second input tuple
    :return: True if the tuples differ in exactly one coordinate, False otherwise
    """
    return len(tuple1) == len(tuple2) and sum(x != y for x, y in zip(tuple1, tuple2)) == 1


def sparse_diag(data: jnp.ndarray, size: int) -> sparse.BCOO:
    range_n = jnp.arange(size)
    indices = jnp.stack([range_n, range_n], axis=1)

    # --- Create BCOO Matrix ---
    return sparse.BCOO((data, indices), shape=(size, size))


def jax_gamma_logic_func_builder(ips, src, tgt, root_state, one_state, root_type=None,
                                 one_type=None, root_one_type=None):
    """
    Returns metadata for gamma calculation.
    [(weight, rate_args, state_index), ...]
    where rate_args is a dict with everything needed to call ips.rate later.
    """

    if ips.vertex_type_space is None and ips.edge_type_space is None:
        return [
            [
                (1 + len(remaining_state)),  # weight
                {  # rate_args (metadata for computing rate later)
                    'src': one_state,
                    'tgt': tgt,
                    'neighbor_states': remaining_state,
                    'neighbors_vertex_type': None,
                    'neighbors_edge_type': None,
                },
                (root_state, one_state) + remaining_state  # state for indexing
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is not None and ips.edge_type_space is None:
        return [
            [
                (1 + len(remaining_state)),
                {
                    'src': one_state,
                    'tgt': tgt,
                    'neighbor_states': remaining_state,
                    'neighbors_vertex_type': (root_type, one_type) + remaining_type,
                    'neighbors_edge_type': None,
                },
                ((root_state, one_state) + remaining_state, (root_type, one_type) + remaining_type)
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.vertex_type_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is None and ips.edge_type_space is not None:
        return [
            [
                (1 + len(remaining_state)),
                {
                    'src': one_state,
                    'tgt': tgt,
                    'neighbor_states': remaining_state,
                    'neighbors_vertex_type': None,
                    'neighbors_edge_type': (root_one_type,) + remaining_type,
                },
                ((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.edge_type_space, repeat=k - 1)
        ]


def jax_build_static_maps(ips, ode_state_space, vertex_state_space, state_to_index, ode_state_to_index,
                          gamma_logic_func, vertex_type_space=None, edge_type_space=None):
    """
    Pre-computes static structure for JAX.
    OPTIMIZED: Converts states to Integers IMMEDIATELY to prevent OOM errors.
    """

    rows = []
    cols = []
    root_jump_indices = []
    neighbor_jump_indices = []

    # Root jump data
    root_src_list = []
    root_tgt_list = []
    root_neighbor_states_list = []  # Will store lists of INTs
    root_neighbor_vertex_types_list = []
    root_neighbor_edge_types_list = []

    # Gamma data structures
    gamma_dependency_indices = []
    gamma_weights = []

    # Flattened gamma term data - STORE INTEGERS HERE
    gamma_src_list = []
    gamma_tgt_list = []
    gamma_neighbor_states_list = []  # List of lists of INTs
    gamma_neighbor_vertex_types_list = []
    gamma_neighbor_edge_types_list = []

    current_flat_index = 0
    gamma_gather_indices_list = []
    transition_idx = -1

    # Pre-compute type maps if they exist to allow immediate conversion
    v_to_i = {}
    if ips.vertex_type_space is not None:
        v_to_i = {vt: i for i, vt in enumerate(ips.vertex_type_space)}

    e_to_i = {}
    if ips.edge_type_space is not None:
        e_to_i = {et: i for i, et in enumerate(ips.edge_type_space)}

    # --- MAIN LOOP ---
    for src, tgt in product(vertex_state_space, repeat=2):
        if one_coordinate_apart(src, tgt):

            # [Type space setup remains same]
            type_space = ['empty']
            if ips.vertex_type_space is not None and ips.edge_type_space is None:
                type_space = vertex_type_space
            elif ips.vertex_type_space is None and ips.edge_type_space is not None:
                type_space = edge_type_space

            for neighbor_types in type_space:
                # [Filter logic remains same]
                if (neighbor_types == 'empty' and ips.vertex_type_space is None and ips.edge_type_space is None) or \
                        (ips.vertex_type_space is not None and ips.edge_type_space is None and len(
                            neighbor_types) == len(src)) or \
                        (ips.vertex_type_space is None and ips.edge_type_space is not None and len(
                            neighbor_types) == len(src) - 1):

                    if neighbor_types == 'empty': neighbor_types = ()
                    transition_idx += 1

                    # Store Matrix Indices
                    if ips.vertex_type_space is None and ips.edge_type_space is None:
                        rows.append(ode_state_to_index[src])
                        cols.append(ode_state_to_index[tgt])
                    else:
                        rows.append(ode_state_to_index[(src, neighbor_types)])
                        cols.append(ode_state_to_index[(tgt, neighbor_types)])

                    # --- ROOT JUMP ---
                    if src[0] != tgt[0]:
                        root_jump_indices.append(transition_idx)

                        # Store Integers Immediately
                        root_src_list.append(state_to_index[src[0]])
                        root_tgt_list.append(state_to_index[tgt[0]])

                        # Neighbors (Prepend Root)
                        if ips.vertex_type_space is not None:
                            # Store (Root, N1...) as integers
                            neighs = (src[0],) + src[1:]
                            root_neighbor_states_list.append([state_to_index[s] for s in neighs])

                            # Store Types as integers
                            root_neighbor_vertex_types_list.append([v_to_i[v] for v in neighbor_types])
                        else:
                            neighs = src[1:]
                            root_neighbor_states_list.append([state_to_index[s] for s in neighs])
                            root_neighbor_vertex_types_list.append([-1])  # Dummy

                        # Edge Types
                        if ips.edge_type_space is not None:
                            root_neighbor_edge_types_list.append([e_to_i[e] for e in neighbor_types])
                        else:
                            root_neighbor_edge_types_list.append([-1])

                    # --- NEIGHBOR JUMP ---
                    else:
                        neighbor_jump_indices.append(transition_idx)
                        changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

                        needed_terms = gamma_logic_func(
                            ips, src[changed_index], tgt[changed_index],
                            src[changed_index], src[0],
                            root_type=neighbor_types[changed_index] if ips.vertex_type_space is not None else None,
                            one_type=neighbor_types[0] if ips.vertex_type_space is not None else None,
                            root_one_type=neighbor_types[0] if ips.edge_type_space is not None else None
                        )

                        # Flatten Data
                        count = len(needed_terms)
                        indices = list(range(current_flat_index, current_flat_index + count))
                        gamma_gather_indices_list.append(indices)
                        current_flat_index += count

                        # Store Dependency indices (Already integers via ode_state_to_index)
                        term_indices = [ode_state_to_index[s] for w, r, s in needed_terms]
                        term_weights = [w for w, r, s in needed_terms]
                        gamma_dependency_indices.append(term_indices)
                        gamma_weights.append(term_weights)

                        # --- MEMORY OPTIMIZATION ---
                        # Convert strings to INTs immediately
                        for w, rate_args, s in needed_terms:
                            gamma_src_list.append(state_to_index[rate_args['src']])
                            gamma_tgt_list.append(state_to_index[rate_args['tgt']])

                            # Convert Tuple of Strings -> List of Ints
                            g_neighs = rate_args['neighbor_states']
                            gamma_neighbor_states_list.append([state_to_index[n] for n in g_neighs])

                            # Types
                            if rate_args['neighbors_vertex_type']:
                                gamma_neighbor_vertex_types_list.append(
                                    [v_to_i[v] for v in rate_args['neighbors_vertex_type']])
                            else:
                                gamma_neighbor_vertex_types_list.append([-1])

                            if rate_args['neighbors_edge_type']:
                                gamma_neighbor_edge_types_list.append(
                                    [e_to_i[e] for e in rate_args['neighbors_edge_type']])
                            else:
                                gamma_neighbor_edge_types_list.append([-1])

    # --- PADDING (Lists are now Integers) ---

    # 1. Trash Can Data
    safety_index = len(gamma_src_list)
    gamma_src_list.append(0)
    gamma_tgt_list.append(0)
    gamma_neighbor_states_list.append([-1])
    gamma_neighbor_vertex_types_list.append([-1])
    gamma_neighbor_edge_types_list.append([-1])

    # 2. Pad Root Arrays (Fast Copy)
    num_root_jumps = len(root_src_list)

    max_root_states = max((len(x) for x in root_neighbor_states_list), default=0)
    max_root_vtypes = max((len(x) for x in root_neighbor_vertex_types_list), default=0)
    max_root_cols = max(max_root_states, max_root_vtypes)

    root_neighbors_padded = np.full((num_root_jumps, max_root_cols), -1, dtype=np.int32)
    for i, ns in enumerate(root_neighbor_states_list):
        root_neighbors_padded[i, :len(ns)] = ns  # ns is already [int, int...]

    root_vtypes_padded = np.full((num_root_jumps, max_root_cols), -1, dtype=np.int32)
    if ips.vertex_type_space is not None:
        for i, vt in enumerate(root_neighbor_vertex_types_list):
            root_vtypes_padded[i, :len(vt)] = vt  # vt is already [int, int...]

    root_etypes_padded = np.full((num_root_jumps, max_root_cols), -1, dtype=np.int32)
    if ips.edge_type_space is not None:
        for i, et in enumerate(root_neighbor_edge_types_list):
            root_etypes_padded[i, :len(et)] = et

    # 3. Pad Gamma Arrays (Fast Copy)
    num_gamma_terms = len(gamma_src_list)

    max_gamma_states = max((len(x) for x in gamma_neighbor_states_list), default=0)
    max_gamma_vtypes = max((len(x) for x in gamma_neighbor_vertex_types_list), default=0)
    max_gamma_cols = max(max_gamma_states, max_gamma_vtypes)

    gamma_neighbors_padded = np.full((num_gamma_terms, max_gamma_cols), -1, dtype=np.int32)
    for i, ns in enumerate(gamma_neighbor_states_list):
        gamma_neighbors_padded[i, :len(ns)] = ns  # ns is already ints

    gamma_vtypes_padded = np.full((num_gamma_terms, max_gamma_cols), -1, dtype=np.int32)
    if ips.vertex_type_space is not None:
        for i, vt in enumerate(gamma_neighbor_vertex_types_list):
            gamma_vtypes_padded[i, :len(vt)] = vt

    gamma_etypes_padded = np.full((num_gamma_terms, max_gamma_cols), -1, dtype=np.int32)
    if ips.edge_type_space is not None:
        for i, et in enumerate(gamma_neighbor_edge_types_list):
            gamma_etypes_padded[i, :len(et)] = et

    # 4. Pad Gather Map
    num_neigh_jumps = len(neighbor_jump_indices)
    max_gather_terms = max((len(x) for x in gamma_gather_indices_list), default=0)

    gamma_gather_map = np.full((num_neigh_jumps, max_gather_terms), safety_index, dtype=np.int32)
    for i, inds in enumerate(gamma_gather_indices_list):
        gamma_gather_map[i, :len(inds)] = inds

    # 5. Pad Weights & Indices
    gamma_indices_padded = np.zeros((num_neigh_jumps, max_gather_terms), dtype=np.int32)
    gamma_weights_padded = np.zeros((num_neigh_jumps, max_gather_terms), dtype=np.float32)

    for i in range(num_neigh_jumps):
        n = len(gamma_dependency_indices[i])
        gamma_indices_padded[i, :n] = gamma_dependency_indices[i]
        gamma_weights_padded[i, :n] = gamma_weights[i]

    # Bundle
    static_args = {
        "rows": jnp.array(rows, dtype=jnp.int32),
        "cols": jnp.array(cols, dtype=jnp.int32),
        "root_idx_map": jnp.array(root_jump_indices, dtype=jnp.int32),

        # Already INTs
        "root_src": jnp.array(root_src_list, dtype=jnp.int32),
        "root_tgt": jnp.array(root_tgt_list, dtype=jnp.int32),
        "neighbors": jnp.array(root_neighbors_padded, dtype=jnp.int32),
        "neighbors_vertex_types": jnp.array(root_vtypes_padded, dtype=jnp.int32),
        "neighbors_edge_types": jnp.array(root_etypes_padded, dtype=jnp.int32),

        "neigh_idx_map": jnp.array(neighbor_jump_indices, dtype=jnp.int32),
        "gamma_indices": jnp.array(gamma_indices_padded, dtype=jnp.int32),
        "gamma_weights": jnp.array(gamma_weights_padded, dtype=jnp.float32),
        "gamma_gather_map": jnp.array(gamma_gather_map, dtype=jnp.int32),

        # Already INTs
        "gamma_src": jnp.array(gamma_src_list, dtype=jnp.int32),
        "gamma_tgt": jnp.array(gamma_tgt_list, dtype=jnp.int32),
        "gamma_neighbors": jnp.array(gamma_neighbors_padded, dtype=jnp.int32),
        "gamma_neighbors_vertex_types": jnp.array(gamma_vtypes_padded, dtype=jnp.int32),
        "gamma_neighbors_edge_types": jnp.array(gamma_etypes_padded, dtype=jnp.int32),

        "num_states": len(ode_state_space),
    }

    # Don't forget global map if using it
    # ...

    sparse_indices = jnp.stack([static_args["rows"], static_args["cols"]], axis=1)
    return static_args, sparse_indices


def jax_mlfe_vector_field(t, p, args):
    """
    Fully vectorized vector field computation.

    This function is JIT-compiled by diffrax.diffeqsolve.
    All operations must be JAX-compatible (no Python control flow on traced values).

    Args:
        t: scalar time
        p: (num_states,) array of probabilities
        args: dict containing static structure and rate caller

    Returns:
        (num_states,) array of dp/dt
    """

    rate_caller = args["rate_caller"]

    # -------------------------------------------------
    # 1. Compute all root jump rates (vectorized)
    # -------------------------------------------------
    root_rates = rate_caller(
        args["root_src"],
        args["root_tgt"],
        args["neighbors"],
        args["neighbors_vertex_types"],
        args["neighbors_edge_types"],
        p
    )

    # -------------------------------------------------
    # 2. Compute all gamma term rates (vectorized)
    # -------------------------------------------------
    # Call rate function for ALL flattened gamma terms at once
    all_gamma_term_rates = rate_caller(
        args["gamma_src"],
        args["gamma_tgt"],
        args["gamma_neighbors"],
        args["gamma_neighbors_vertex_types"],
        args["gamma_neighbors_edge_types"],
        p
    )

    # -------------------------------------------------
    # 3. Assemble gamma rates for each neighbor jump
    # -------------------------------------------------
    # Gather probabilities for all terms
    term_probs = p[args["gamma_indices"]]  # (num_neighbor_jumps, max_terms)

    gamma_rates = all_gamma_term_rates[args["gamma_gather_map"]]

    # Calculate gamma: rate = sum(weight * prob * rate) / sum(weight * prob)
    denom_terms = term_probs * args["gamma_weights"]
    denoms = jnp.sum(denom_terms, axis=1)

    num_terms = denom_terms * gamma_rates
    nums = jnp.sum(num_terms, axis=1)

    neigh_rates = nums / (denoms + 1e-12)  # Add epsilon to avoid division by zero

    # -------------------------------------------------
    # 4. Assemble full rate matrix
    # -------------------------------------------------
    total_transitions = len(args["rows"])
    all_rates = jnp.zeros(total_transitions, dtype=jnp.float32)

    # Scatter rates into full array
    all_rates = all_rates.at[args["neigh_idx_map"]].set(neigh_rates)
    all_rates = all_rates.at[args["root_idx_map"]].set(root_rates)

    # Build sparse rate matrix Q
    sparse_indices = jnp.stack([args["rows"], args["cols"]], axis=1)
    Q_off = sparse.BCOO(
        (all_rates, sparse_indices),
        shape=(args["num_states"], args["num_states"])
    )

    # Adjust diagonal: Q[i,i] = -sum of row i (excluding diagonal)
    row_sums = sparse.bcoo_reduce_sum(Q_off, axes=[1]).todense().reshape(-1)

    # Create diagonal sparse matrix
    diag_indices = jnp.stack([jnp.arange(args["num_states"]), jnp.arange(args["num_states"])], axis=1)
    Q_diag = sparse.BCOO(
        (row_sums, diag_indices),
        shape=(args["num_states"], args["num_states"])
    )

    Q_final = Q_off - Q_diag

    # Return dp/dt = Q^T @ p
    return Q_final.T @ p


def make_rate_caller(rate_func_vectorized, params, has_vertex_types, has_edge_types):
    """
    Creates a wrapper around the user's rate function that handles:
    - Parameter passing
    - Masking padded values
    - Type handling
    """
    @jax.jit
    def call_rates(src, tgt, neighbors, vertex_types, edge_types, meas):
        """
        Vectorized rate computation.

        Args:
            src: (N,) array of source states
            tgt: (N,) array of target states
            neighbors: (N, max_neighbors) array of neighbor states (padded with -1) TODO: update max_neighbors + 1
            vertex_types: (N, max_neighbors) array of vertex types (padded with -1)
            edge_types: (N, max_neighbors) array of edge types (padded with -1)
            has_vertex_types: bool
            has_edge_types: bool

        Returns:
            (N,) array of rates
        """
        # Mask out padded neighbors (-1 values)
        # This ensures padded values don't affect sums/counts
        valid_mask = neighbors >= 0
        masked_neighbors = jnp.where(valid_mask, neighbors, 0)

        # Call user's rate function
        if has_vertex_types:
            masked_vertex_types = jnp.where(valid_mask, vertex_types, 0)
            rates = jax.vmap(
                lambda s, t, n, vt, p: rate_func_vectorized(
                    s, t, n, vertex_types=vt, params=params, meas=p
                ),
                in_axes=(0, 0, 0, 0, None)
            )(src, tgt, masked_neighbors, masked_vertex_types, meas)
        elif has_edge_types:
            masked_edge_types = jnp.where(valid_mask, edge_types, 0)
            rates = jax.vmap(
                lambda s, t, n, et, p: rate_func_vectorized(
                    s, t, n, edge_types=et, params=params, meas=p
                ),
                in_axes=(0, 0, 0, 0, None)
            )(src, tgt, masked_neighbors, masked_edge_types, meas)
        else:
            rates = jax.vmap(
                lambda s, t, n, p: rate_func_vectorized(s, t, n, params=params, meas=p),
                in_axes=(0, 0, 0, None)
            )(src, tgt, masked_neighbors, meas)

        return rates

    return call_rates


def simulate_markov_lfe(
        ips: ParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        vertex_type_init: dict[any, float] = None,
        edge_type_init: dict[any, float] = None,
        num_grid_points: int = 100,
        jit: str = 'jax',
        solver_type: str = 'explicit',
        step_control: str = 'constant',
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[any]]]:
    # [Keep all existing setup code for deg_dist, state spaces, etc.]
    deg_dist = ips.get_empirical_degree_distribution()
    deg_supp = [i for (i, p) in deg_dist.items() if p > 0]

    # [Keep vertex_state_space and ode_state_space construction - same as before]
    if ips.vertex_type_space is None and ips.edge_type_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        ode_state_space = vertex_state_space
    elif ips.vertex_type_space is not None and ips.edge_type_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        vertex_type_space = [(root,) + children for k in deg_supp for (root, children) in
                             product(ips.vertex_type_space, product(ips.vertex_type_space, repeat=k))]
        ode_state_space = [(state, type) for state in vertex_state_space for type in vertex_type_space if
                           len(state) == len(type)]
    elif ips.vertex_type_space is None and ips.edge_type_space is not None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        edge_type_space = [root_children for k in deg_supp for root_children in product(ips.edge_type_space, repeat=k)]
        ode_state_space = [(state, type) for state in vertex_state_space for type in edge_type_space if
                           len(state) == len(type) + 1]

    index_to_ode_state_space = {i: state for i, state in enumerate(ode_state_space)}
    ode_state_space_to_index = {state: i for i, state in enumerate(ode_state_space)}

    # [Keep initial condition calculation - same as before]
    if ips.vertex_type_space is None and ips.edge_type_space is None:
        ode_init = [initial_conditions[state[0]] * deg_dist[len(state) - 1] * np.prod(
            [initial_conditions[child] for child in state[1:]]) for state in ode_state_space]
    elif ips.vertex_type_space is not None and ips.edge_type_space is None:
        ode_init = [
            initial_conditions[state[0]]
            * deg_dist[len(state) - 1]
            * np.prod([initial_conditions[child] for child in state[1:]])
            * np.prod([vertex_type_init[t] for t in type])
            for (state, type) in ode_state_space
        ]
    elif ips.vertex_type_space is None and ips.edge_type_space is not None:
        ode_init = [
            initial_conditions[state[0]]
            * deg_dist[len(state) - 1]
            * np.prod([initial_conditions[child] for child in state[1:]])
            * np.prod([edge_type_init[t] for t in type])
            for (state, type) in ode_state_space
        ]

    if jit == 'jax':
        # Validate that user provided vectorized rate function
        if not hasattr(ips, 'rate_vectorized'):
            raise ValueError(
                "JAX backend requires 'rate_vectorized' method. "
                "This method must use JAX-compatible operations (jnp.where, jax.vmap, etc.) "
                "and accept vectorized inputs. See documentation for examples."
            )

        # Extract parameters from ips
        if ips.params is None:
            raise ValueError("JAX backend requires 'params' attribute in ParticleSystem.")
        rate_params = ips.params

        # Build static structure
        print('**** Building rate matrix structure ****')
        static_args, sparse_indices = jax_build_static_maps(
            ips, ode_state_space, vertex_state_space, ips.get_state_to_index_map(), ode_state_space_to_index,
            jax_gamma_logic_func_builder,
            vertex_type_space=vertex_type_space if ips.vertex_type_space is not None else None,
            edge_type_space=edge_type_space if ips.edge_type_space is not None else None
        )

        # Create rate caller
        rate_caller = make_rate_caller(ips.rate_vectorized,
                                       rate_params,
                                       ips.vertex_type_space is not None,
                                       ips.edge_type_space is not None)
        static_args["rate_caller"] = rate_caller

        # Initialize
        y0 = jnp.array(ode_init)

        # Define ODE term
        term = diffrax.ODETerm(jax_mlfe_vector_field)

        # Choose solver
        if solver_type == 'implicit':
            linear_solver = lx.GMRES(rtol=1e-3, atol=1e-3, restart=20)
            root_finder = optx.Newton(rtol=1e-3, atol=1e-3, linear_solver=linear_solver)
            solver = diffrax.Kvaerno3(root_finder=root_finder)
        elif solver_type == 'explicit':
            solver = diffrax.Dopri5()
        else:
            raise ValueError(f'Unknown solver type: {solver_type}')

        # Define output times
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, max_time, num_grid_points))

        # Choose step controller
        if step_control == 'adaptive':
            step_controller = diffrax.PIDController(rtol=1e-3, atol=1e-9)
        elif step_control == 'constant':
            step_controller = diffrax.ConstantStepSize()
        else:
            raise ValueError(f'Unknown step control type: {step_control}')

        print('**** Running simulation ****')
        gc.collect()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=max_time,
            dt0=0.01,
            y0=y0,
            args=static_args,
            stepsize_controller=step_controller,
            saveat=saveat,
            max_steps=100000,
            progress_meter=diffrax.TqdmProgressMeter()
        )

        return sol.ts, sol.ys.transpose(), index_to_ode_state_space

    else:
        # Original non-JAX implementation
        pass


def simulate_markov_lfe_mf(
        ips: ParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        num_particles: int = 500,
        seed: int = 42,
        num_grid_points: int = 100,
        gamma: callable = None
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[any]]]:
    # model parameters
    deg_dist = ips.get_empirical_degree_distribution()
    deg_supp = [i for (i, p) in deg_dist.items() if p > 0]

    # track all possible root-children marginals
    vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                          product(ips.state_space, product(ips.state_space, repeat=k))]
    ode_state_space = [(root,) + children for k in deg_supp for (root, children) in
                       product(ips.state_space, product(ips.state_space, repeat=k))]

    index_to_ode_state_space = {i: state for i, state in enumerate(ode_state_space)}
    ode_state_space_to_index = {state: i for i, state in index_to_ode_state_space.items()}

    if gamma is None:
        # define the Markov local-field jump rate
        def gamma(src: tuple, tgt: tuple, root_state, one_state, marginal_prob: dict[tuple, float]) -> float:
            numerator = sum(
                (2 + len(remaining_state)) *
                marginal_prob.get((root_state, one_state) + remaining_state, 0) * ips.rate(src, tgt, (
                    one_state,) + remaining_state)
                for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
            )
            denominator = sum(
                (2 + len(remaining_state)) * marginal_prob.get((root_state, one_state) + remaining_state, 0)
                for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
            )
            return numerator / denominator if denominator > 0 else 0

    class MLFEParticleSystem(MeanFieldParticleSystem):
        def __init__(self, ode_state_space: list[any], num_particles: int, name: str = None):
            super().__init__(state_space=ode_state_space, num_particles=num_particles, name=name)

        def rate(self, src: any,
                 tgt: any,
                 meas: dict[tuple[any], float]) -> float:
            if one_coordinate_apart(src, tgt):
                # find the index of the changed coordinate
                changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

                # if the root jumped, return usual rate
                if changed_index == 0:
                    return ips.rate(src[0], tgt[0], src[1:])
                # otherwise, take conditional rate
                else:
                    return gamma(src[changed_index], tgt[changed_index], src[changed_index], src[0], meas)
            return 0.0

    # calculate initial conditions on ode_state_space given i.i.d. initial conditions on vertices
    mf_init = {}
    for i in range(num_particles):
        # pick offspring number from degree distribution
        k = np.random.choice(deg_supp, p=[deg_dist[i] for i in deg_supp])
        # pick states for root and k leafs
        init_state = tuple(
            [np.random.choice(ips.state_space, p=[initial_conditions[s] for s in ips.state_space]) for _ in
             range(k + 1)])
        # add to initial conditions
        mf_init[i] = init_state

    # create the mean-field particle system
    mfps = MLFEParticleSystem(ode_state_space, num_particles, name=f"Mean-field approximation for MLFE for {ips.name}")
    # simulate the mean-field particle system (timed)
    mf_jump_list = simulate_mean_field_jump_process(mfps=mfps, initial_conditions=mf_init, max_time=max_time, seed=seed)

    # convert jump list to array with time and state
    time_points = np.linspace(0, max_time, num_grid_points)
    time_state_dict = get_particle_states_at_times(mf_jump_list, mf_init, list(time_points))

    time_state_ndarray = np.zeros((len(ode_state_space), num_grid_points))
    for t_idx, state_at_time in enumerate(time_state_dict):
        counter = Counter(state_at_time.values())
        for state in ode_state_space:
            time_state_ndarray[ode_state_space_to_index[state], t_idx] = counter[state] / num_particles

    return time_points, time_state_ndarray, index_to_ode_state_space
