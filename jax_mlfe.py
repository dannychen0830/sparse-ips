from itertools import product
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

### This file contains all JAX-specific functions for simulating Markov LFE


###########################
# JIT-related helpers
###########################
def compute_ode_state_to_index(ips):
    deg_supp = ips.deg_supp

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

    ode_state_space_to_index = {state: i for i, state in enumerate(ode_state_space)}

    return ode_state_space_to_index


def compute_jax_static_args(ips):
    deg_supp = ips.deg_supp

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

    ode_state_space_to_index = {state: i for i, state in enumerate(ode_state_space)}

    static_args, sparse_indices = jax_build_static_maps_vmap(
        ips, ode_state_space, vertex_state_space, ips.get_state_to_index_map(), ode_state_space_to_index,
        jax_gamma_index_builder_vmap,
        vertex_type_space=vertex_type_space if ips.vertex_type_space is not None else None,
        edge_type_space=edge_type_space if ips.edge_type_space is not None else None
    )

    return ode_state_space_to_index, static_args, sparse_indices


def make_rate_caller(rate_func_vectorized, params, has_vertex_types, has_edge_types, has_edge_states):
    """
    Creates a wrapper around the user's rate function that handles:
    - Parameter passing
    - Masking padded values
    - Type handling
    """
    @jax.jit
    def call_rates(src, tgt, neighbors, vertex_types, edge_types, edge_states, meas, t):
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
        # Call user's rate function
        if has_vertex_types:
            rates = jax.vmap(
                lambda src, tgt, nei, vt, p, t: rate_func_vectorized(
                    src, tgt, nei, vertex_types=vt, params=params, meas=p, t=t
                ),
                in_axes=(0, 0, 0, 0, None, None)
            )(src, tgt, neighbors, vertex_types, meas, t)
        elif has_edge_types:
            rates = jax.vmap(
                lambda src, tgt, nei, et, p, t: rate_func_vectorized(
                    src, tgt, nei, edge_types=et, params=params, meas=p, t=t
                ),
                in_axes=(0, 0, 0, 0, None, None)
            )(src, tgt, neighbors, edge_types, meas, t)
        else:
            rates = jax.vmap(
                lambda src, tgt, nei, p, t: rate_func_vectorized(src, tgt, nei, params=params, meas=p, t=t),
                in_axes=(0, 0, 0, None, None)
            )(src, tgt, neighbors, meas, t)

        return rates

    return call_rates


def make_edge_rate_caller(rate_func_vectorized, params):
    @jax.jit
    def call_rates(src, tgt, neighbors, meas, t):
        rates = jax.vmap(
            lambda src, tgt, nei, p, t: rate_func_vectorized(src, tgt, nei, params=params, meas=p, t=t),
            in_axes=(0, 0, 0, None, None)
        )(src, tgt, neighbors, meas, t)

        return rates

    return call_rates


###########################
###########################


###########################
# general helper functions
###########################
def x_coordinate_apart(tuple1: tuple, tuple2: tuple, x: int = 1) -> bool:
    """ 
    Check if two tuples are one coordinate apart, i.e., they differ in exactly one coordinate.
    :param tuple1: first input tuple
    :param tuple2: second input tuple
    :param x: the coordinate to check
    :return: True if the tuples differ in exactly x coordinate, False otherwise
    """
    return len(tuple1) == len(tuple2) and sum(x != y for x, y in zip(tuple1, tuple2)) == x


###########################
###########################


###########################
# indexing without global dependency (precompute rate)
###########################

def jax_gamma_index_builder(ips, src, tgt, root_state, one_state, root_type=None, one_type=None, root_one_type=None):
    # print('****warning: global interaction is not considered in this function****')

    if ips.vertex_type_space is None and ips.edge_type_space is None:
        return [
            [(1 + len(remaining_state)),
             ips.rate(src, tgt, (one_state,) + remaining_state),
             (root_state, one_state) + remaining_state]
            for k in ips.deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is not None and ips.edge_type_space is None:
        return [
            [(1 + len(remaining_state)),
             ips.rate(src, tgt, (one_state,) + remaining_state,
                      neighbors_vertex_type=(root_type, one_type) + remaining_type),
             ((root_state, one_state) + remaining_state, (root_type, one_type) + remaining_type)]

            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.vertex_type_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is None and ips.edge_type_space is not None:
        return [
            [(1 + len(remaining_state)),
             ips.rate(src, tgt, (one_state,) + remaining_state, neighbors_edge_type=(root_one_type,) + remaining_type),
             ((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.edge_type_space, repeat=k - 1)
        ]


def jax_build_static_maps(ips, ode_state_space, vertex_state_space, ode_state_to_index, gamma_logic_func,
                      vertex_type_space=None, edge_type_space=None):
    """
    Analyzes the graph structure and returns static arrays for JAX.
    """
    rows = []
    cols = []

    # Lists to store  needed for Root Jumps vs Neighbor Jumps
    # We split them because they have different rate calculations
    root_jump_indices = []  # Indices in 'rows' that are root jumps
    neighbor_jump_indices = []  # Indices in 'rows' that are neighbor jumps (use gamma)

    root_rates = []

    # Data for Gamma Calculation (Neighbor Jumps)
    # We need a list of lists, where each sublist contains the p-indices
    # needed to compute the sum for that specific transition.
    gamma_dependency_indices = []
    gamma_weights = []
    gamma_rates = []



    transition_idx = -1
    for src, tgt in product(vertex_state_space, repeat=2):

        if x_coordinate_apart(src, tgt):
            type_space = ['empty']
            if ips.vertex_type_space is not None and ips.edge_type_space is None:
                type_space = vertex_type_space
            elif ips.vertex_type_space is None and ips.edge_type_space is not None:
                type_space = edge_type_space

            for neighbor_types in type_space:
                if (neighbor_types =='empty' and ips.vertex_type_space is None and ips.edge_type_space is None) or \
                        (ips.vertex_type_space is not None and ips.edge_type_space is None and len(neighbor_types) == len(src)) or \
                        (ips.vertex_type_space is None and ips.edge_type_space is not None and len(neighbor_types) == len(src) - 1):

                    if neighbor_types == 'empty':
                        neighbor_types = ()

                    transition_idx += 1

                    if ips.vertex_type_space is None and ips.edge_type_space is None:
                        row_idx = ode_state_to_index[src]
                        col_idx = ode_state_to_index[tgt]
                    else:
                        row_idx = ode_state_to_index[(src, neighbor_types)]
                        col_idx = ode_state_to_index[(tgt, neighbor_types)]

                    rows.append(row_idx)
                    cols.append(col_idx)

                    if src[0] != tgt[0]:
                        root_jump_indices.append(transition_idx)
                        root_rates.append(ips.rate(src[0], tgt[0], src[1:],
                                                   neighbors_vertex_type=neighbor_types if ips.vertex_type_space is not None else None,
                                                   neighbors_edge_type=neighbor_types if ips.edge_type_space is not None else None))
                    # leaf jump
                    else:
                        neighbor_jump_indices.append(transition_idx)

                        # compute tuples (weight, rate, state) needed for gamma calculation
                        changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])
                        needed_terms = gamma_logic_func(ips, src[changed_index], tgt[changed_index],
                                                        src[changed_index], src[0],
                                                        root_type=neighbor_types[changed_index] if ips.vertex_type_space is not None else None,
                                                        one_type=neighbor_types[0] if ips.vertex_type_space is not None else None,
                                                        root_one_type=neighbor_types[changed_index-1] if ips.edge_type_space is not None else None)

                        term_indices = [ode_state_to_index[s] for w, r, s in needed_terms]
                        term_rates = [r for w, r, s in needed_terms]
                        term_weights = [w for w, r, s in needed_terms]

                        gamma_dependency_indices.append(term_indices)
                        gamma_weights.append(term_weights)
                        gamma_rates.append(term_rates)

    # Find max number of terms in any gamma sum
    max_terms = max(len(x) for x in gamma_dependency_indices) if gamma_dependency_indices else 0

    # Create padded arrays
    num_neighbor_jumps = len(neighbor_jump_indices)
    padded_indices = np.zeros((num_neighbor_jumps, max_terms), dtype=np.int32)
    padded_weights = np.zeros((num_neighbor_jumps, max_terms), dtype=np.float64)
    padded_rates = np.zeros((num_neighbor_jumps, max_terms), dtype=np.float64)

    for i in range(num_neighbor_jumps):
        terms = gamma_dependency_indices[i]
        padded_indices[i, :len(terms)] = terms
        padded_weights[i, :len(terms)] = gamma_weights[i]
        padded_rates[i, :len(terms)] = gamma_rates[i]
        # Remaining slots are 0 index and 0.0 weight (no effect)

    # Bundle everything into a struct or dict
    static_args = {
        "rows": jnp.array(rows),
        "cols": jnp.array(cols),
        "root_idx_map": jnp.array(root_jump_indices),
        "root_rates": jnp.array(root_rates),
        "neigh_idx_map": jnp.array(neighbor_jump_indices),
        "gamma_indices": jnp.array(padded_indices),
        "gamma_rates": jnp.array(padded_rates),
        "gamma_weights": jnp.array(padded_weights),
        "num_states": len(ode_state_space)
    }

    # Also return the fixed structure for the sparse matrix
    sparse_indices = jnp.stack([static_args["rows"], static_args["cols"]], axis=1)

    return static_args, sparse_indices

###########################
###########################


###########################
# indexing with global dependency (jit and vectorize rate)
###########################

def jax_gamma_index_builder_vmap(ips, src, tgt, root_state, one_state, root_type=None,
                                 one_type=None, root_one_type=None, root_one_state=None):
    """
    Returns metadata for gamma calculation.
    [(weight, rate_args, state_index), ...]
    where rate_args is a dict with everything needed to call ips.rate later.
    """

    if ips.vertex_type_space is None and ips.edge_type_space is None and root_one_state is None:
        return [
            [
                (1 + len(remaining_state)),  # weight
                {  # rate_args (metadata for computing rate later)
                    'src': src,
                    'tgt': tgt,
                    'neighbor_states': (one_state,) + remaining_state,
                    'neighbors_vertex_type': None,
                    'neighbors_edge_type': None,
                    'neighbors_edge_state': None,
                },
                (root_state, one_state) + remaining_state  # state for indexing
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is not None and ips.edge_type_space is None and root_one_state is None:
        return [
            [
                (1 + len(remaining_state)),
                {
                    'src': src,
                    'tgt': tgt,
                    'neighbor_states': (one_state,) + remaining_state,
                    'neighbors_vertex_type': (root_type, one_type) + remaining_type,
                    'neighbors_edge_type': None,
                    'neighbors_edge_state': None,
                },
                ((root_state, one_state) + remaining_state, (root_type, one_type) + remaining_type)
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.vertex_type_space, repeat=k - 1)
        ]

    elif ips.vertex_type_space is None and ips.edge_type_space is not None and root_one_state is None:
        return [
            [
                (1 + len(remaining_state)),
                {
                    'src': src,
                    'tgt': tgt,
                    'neighbor_states': (one_state,) + remaining_state,
                    'neighbors_vertex_type': None,
                    'neighbors_edge_type': (root_one_type,) + remaining_type,
                    'neighbors_edge_state': None,
                },
                ((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_type in product(ips.edge_type_space, repeat=k - 1)
        ]
    
    elif ips.vertex_type_space is None and ips.edge_type_space is None and root_one_state is not None:
        return [
            [
                (1 + len(remaining_state)),
                {
                    'src': one_state,
                    'tgt': tgt,
                    'neighbor_states': remaining_state,
                    'neighbors_vertex_type': None,
                    'neighbors_edge_type': None,
                    'neighbors_edge_state': (root_one_state,) + remaining_edge_state,
                },
                ((root_state, one_state) + remaining_state, (root_one_state,) + remaining_edge_state)
            ]
            for k in ips.deg_supp
            for remaining_state in product(ips.state_space, repeat=k - 1)
            for remaining_edge_state in product(ips.edge_state_space, repeat=k - 1)
        ]


def jax_build_static_maps_vmap(ips, ode_state_space, vertex_state_space, state_to_index, ode_state_to_index,
                          gamma_logic_func, vertex_type_space=None, edge_type_space=None, edge_state_space=None):
    """
    Pre-computes static structure for JAX.
    OPTIMIZED: Converts states to Integers IMMEDIATELY to prevent OOM errors.
    """

    """
        Pre-computes static structure for JAX. Returns arrays that describe:
        - Sparse matrix structure (rows, cols)
        - Which transitions are root vs neighbor jumps
        - Flattened arrays of ALL rate function arguments
        """

    rows = []
    cols = []
    root_jump_indices = []
    neighbor_jump_indices = []

    # Root jump : store as lists then convert to arrays
    root_src_list = []
    root_tgt_list = []
    root_neighbor_states_list = []
    root_neighbor_vertex_types_list = []
    root_neighbor_edge_types_list = []
    root_neighbor_edge_states_list = []

    # Gamma  structures
    gamma_dependency_indices = []
    gamma_weights = []

    # Flattened gamma term 
    gamma_src_list = []
    gamma_tgt_list = []
    gamma_neighbor_states_list = []
    gamma_neighbor_vertex_types_list = []
    gamma_neighbor_edge_types_list = []
    gamma_neighbor_edge_states_list = []

    # Edge jump structures
    edge_jump_indices = []
    edge_src_list = []
    edge_tgt_list = []
    edge_neighbor_vertex_states_list = []

    # build mapping gamma gather map indices
    current_flat_index = 0
    gamma_gather_indices_list = []

    transition_idx = -1

    for src, tgt in product(vertex_state_space, repeat=2):
        if x_coordinate_apart(src, tgt):
            type_space = ['empty']
            if ips.vertex_type_space is not None and ips.edge_type_space is None and ips.edge_state_space is None:
                type_space = vertex_type_space
            elif ips.vertex_type_space is None and ips.edge_type_space is not None and ips.edge_state_space is None:
                type_space = edge_type_space
            elif ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is not None:
                type_space = edge_state_space

            for neighbor_types in type_space:
                if (neighbor_types == 'empty' and ips.vertex_type_space is None and ips.edge_type_space is None) or \
                        (ips.vertex_type_space is not None and ips.edge_type_space is None
                         and len(neighbor_types) == len(src)) or \
                        (ips.vertex_type_space is None and ips.edge_type_space is not None
                         and len(neighbor_types) == len(src) - 1) or \
                        (ips.vertex_type_space is None and ips.edge_type_space is None
                         and len(neighbor_types) == len(src) - 1):

                    if neighbor_types == 'empty':
                        neighbor_types = ()

                    transition_idx += 1

                    if ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is None:
                        row_idx = ode_state_to_index[src]
                        col_idx = ode_state_to_index[tgt]
                    else:
                        row_idx = ode_state_to_index[(src, neighbor_types)]
                        col_idx = ode_state_to_index[(tgt, neighbor_types)]

                    rows.append(row_idx)
                    cols.append(col_idx)

                    # ROOT JUMP
                    if src[0] != tgt[0]:
                        root_jump_indices.append(transition_idx)
                        root_src_list.append(src[0])
                        root_tgt_list.append(tgt[0])
                        root_neighbor_states_list.append(src[1:])
                        root_neighbor_vertex_types_list.append(
                            neighbor_types if ips.vertex_type_space is not None else ()
                        )
                        root_neighbor_edge_types_list.append(
                            neighbor_types if ips.edge_type_space is not None else ()
                        )
                        root_neighbor_edge_states_list.append(
                            neighbor_types if ips.edge_state_space is not None else ()
                        )

                    # NEIGHBOR JUMP
                    else:
                        neighbor_jump_indices.append(transition_idx)
                        changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

                        # Get gamma logic
                        needed_terms = gamma_logic_func(
                            ips, src[changed_index], tgt[changed_index],
                            src[changed_index], src[0],
                            root_type=neighbor_types[changed_index] if ips.vertex_type_space is not None else None,
                            one_type=neighbor_types[0] if ips.vertex_type_space is not None else None,
                            root_one_type=neighbor_types[changed_index-1] if ips.edge_type_space is not None else None,
                            root_one_state=neighbor_types[changed_index-1] if ips.edge_state_space is not None else None
                        )

                        # Extract indices and weights
                        term_indices = [ode_state_to_index[s] for w, rate_args, s in needed_terms]
                        term_weights = [w for w, rate_args, s in needed_terms]

                        gamma_dependency_indices.append(term_indices)
                        gamma_weights.append(term_weights)

                        count = len(needed_terms)
                        indices = list(range(current_flat_index, current_flat_index + count))
                        gamma_gather_indices_list.append(indices)
                        current_flat_index += count

                        # Flatten all rate arguments for vectorized computation
                        for w, rate_args, s in needed_terms:
                            gamma_src_list.append(rate_args['src'])
                            gamma_tgt_list.append(rate_args['tgt'])
                            gamma_neighbor_states_list.append(rate_args['neighbor_states'])
                            gamma_neighbor_vertex_types_list.append(rate_args['neighbors_vertex_type'] or ())
                            gamma_neighbor_edge_types_list.append(rate_args['neighbors_edge_type'] or ())
                            gamma_neighbor_edge_states_list.append(rate_args['neighbors_edge_state'] or ())

    if edge_state_space is not None:
        # gather indices for edge jumps
        for src, tgt in product(edge_state_space, repeat=2):
            if x_coordinate_apart(src, tgt):
                # change indices must have 0
                changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])
                # TODO: this can be optimized as edges are permutation invariant
                for vertex_neighborhood_types in vertex_state_space:
                    transition_idx += 1

                    row_idx = ode_state_to_index[(vertex_neighborhood_types, src)]
                    col_idx = ode_state_to_index[(vertex_neighborhood_types, tgt)]

                    rows.append(row_idx)
                    cols.append(col_idx)
                    edge_jump_indices.append(transition_idx)

                    edge_src_list.append(src[changed_index])
                    edge_tgt_list.append(tgt[changed_index])
                    edge_neighbor_vertex_states_list.append((vertex_neighborhood_types[0], vertex_neighborhood_types[1]))
    

    # Convert lists to padded arrays
    # Root jumps
    max_neighbors = max(ips.deg_supp, default=0) + 1  # Note: + 1 for vertex types, pad the rest
    num_root_jumps = len(root_src_list)

    neighbor_states_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(root_neighbor_states_list):
        neighbor_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    # Similar for types if needed
    root_neighbor_vertex_types_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    if ips.vertex_type_space is not None:
        vertex_type_to_index = {vt: i for i, vt in enumerate(ips.vertex_type_space)}
        for i, nt in enumerate(root_neighbor_vertex_types_list):
            root_neighbor_vertex_types_padded[i, :len(nt)] = [vertex_type_to_index[s] for s in nt]

    root_neighbor_edge_types_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    if ips.edge_type_space is not None:
        edge_type_to_index = {et: i for i, et in enumerate(ips.edge_type_space)}
        for i, nt in enumerate(root_neighbor_edge_types_list):
            root_neighbor_edge_types_padded[i, :len(nt)] = [edge_type_to_index[s] for s in nt]

    root_neighbor_edge_states_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    if ips.edge_state_space is not None:
        edge_state_to_index = {es: i for i, es in enumerate(ips.edge_state_space)}
        for i, es in enumerate(root_neighbor_edge_states_list):
            root_neighbor_edge_states_padded[i, :len(es)] = [edge_state_to_index[s] for s in es]

    # Gamma terms
    num_gamma_terms = len(gamma_src_list)

    gamma_neighbor_states_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(gamma_neighbor_states_list):
        gamma_neighbor_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    gamma_neighbor_vertex_types_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    if ips.vertex_type_space is not None:
        for i, nt in enumerate(gamma_neighbor_vertex_types_list):
            gamma_neighbor_vertex_types_padded[i, :len(nt)] = [vertex_type_to_index[s] for s in nt]

    gamma_neighbor_edge_types_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    if ips.edge_type_space is not None:
        for i, nt in enumerate(gamma_neighbor_edge_types_list):
            gamma_neighbor_edge_types_padded[i, :len(nt)] = [edge_type_to_index[s] for s in nt]
    
    gamma_neighbor_edge_states_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    if ips.edge_state_space is not None:
        for i, es in enumerate(gamma_neighbor_edge_states_list):
            gamma_neighbor_edge_states_padded[i, :len(es)] = [edge_state_to_index[s] for s in es]

    # Pad gamma dependency indices and weights
    max_terms_per_jump = max((len(x) for x in gamma_dependency_indices), default=0)
    num_neighbor_jumps = len(neighbor_jump_indices)

    gamma_indices_padded = np.zeros((num_neighbor_jumps, max_terms_per_jump), dtype=np.int32)
    gamma_weights_padded = np.zeros((num_neighbor_jumps, max_terms_per_jump), dtype=np.float32)

    # Pad the Gather Map
    num_jumps = len(neighbor_jump_indices)
    max_terms_per_jump = max((len(x) for x in gamma_gather_indices_list), default=0)
    gamma_gather_map = np.zeros((num_jumps, max_terms_per_jump), dtype=np.int32)

    for i, inds in enumerate(gamma_gather_indices_list):
        gamma_gather_map[i, :len(inds)] = inds

    for i in range(num_neighbor_jumps):
        n_terms = len(gamma_dependency_indices[i])
        gamma_indices_padded[i, :n_terms] = gamma_dependency_indices[i]
        gamma_weights_padded[i, :n_terms] = gamma_weights[i]

    # Edge jump structures
    num_edge_jumps = len(edge_jump_indices)
    edge_src = np.array([edge_state_to_index[s] for s in edge_src_list], dtype=np.int32)
    edge_tgt = np.array([edge_state_to_index[s] for s in edge_tgt_list], dtype=np.int32)
    edge_neighbor_vertex_states_padded = np.full((num_edge_jumps, 2), -1, dtype=np.int32)
    if ips.edge_state_space is not None:
        for i, ns in enumerate(edge_neighbor_vertex_states_list):
            edge_neighbor_vertex_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    # Bundle everything
    static_args = {
        # Sparse matrix structure
        "rows": jnp.array(rows, dtype=jnp.int32),
        "cols": jnp.array(cols, dtype=jnp.int32),

        # Root jump 
        "root_idx_map": jnp.array(root_jump_indices, dtype=jnp.int32),
        "root_src": jnp.array([state_to_index[s] for s in root_src_list], dtype=jnp.int32),
        "root_tgt": jnp.array([state_to_index[s] for s in root_tgt_list], dtype=jnp.int32),
        "neighbors": jnp.array(neighbor_states_padded, dtype=jnp.int32),
        "neighbors_vertex_types": jnp.array(root_neighbor_vertex_types_padded, dtype=jnp.int32),
        "neighbors_edge_types": jnp.array(root_neighbor_edge_types_padded, dtype=jnp.int32),
        "neighbors_edge_states": jnp.array(root_neighbor_edge_states_padded, dtype=jnp.int32),

        # Neighbor jump 
        "neigh_idx_map": jnp.array(neighbor_jump_indices, dtype=jnp.int32),
        "gamma_indices": jnp.array(gamma_indices_padded, dtype=jnp.int32),
        "gamma_weights": jnp.array(gamma_weights_padded, dtype=jnp.float32),

        # Flattened gamma term  (for vectorized rate calls)
        "gamma_src": jnp.array([state_to_index[s] for s in gamma_src_list], dtype=jnp.int32),
        "gamma_tgt": jnp.array([state_to_index[s] for s in gamma_tgt_list], dtype=jnp.int32),
        "gamma_neighbors": jnp.array(gamma_neighbor_states_padded, dtype=jnp.int32),
        "gamma_neighbors_vertex_types": jnp.array(gamma_neighbor_vertex_types_padded, dtype=jnp.int32),
        "gamma_neighbors_edge_types": jnp.array(gamma_neighbor_edge_types_padded, dtype=jnp.int32),
        "gamma_neighbors_edge_states": jnp.array(gamma_neighbor_edge_states_padded, dtype=jnp.int32),

        # Edge jump structures
        "edge_idx_map": jnp.array(edge_jump_indices, dtype=jnp.int32),
        "edge_src": edge_src,
        "edge_tgt": edge_tgt,
        "edge_neighbor_vertex_states": jnp.array(edge_neighbor_vertex_states_padded, dtype=jnp.int32),

        # Mapping from neighbor jumps to gamma terms
        "gamma_gather_map": jnp.array(gamma_gather_map, dtype=jnp.int32),

        # Metadata
        "num_states": len(ode_state_space),
    }

    sparse_indices = jnp.stack([static_args["rows"], static_args["cols"]], axis=1)
    return static_args, sparse_indices

###########################
###########################


###########################
# indexing with global dependency and edge evolution (jit and vectorize rate)
###########################

def jax_gamma_index_builder_vmap_dynamic_weights(ips, src, tgt, root_state, one_state, root_one_state):
    """
    Returns metadata for gamma calculation.
    [(weight, rate_args, state_index), ...]
    where rate_args is a dict with everything needed to call ips.rate later.
    """

    return [
        [
            (1 + len(remaining_state)),
            {
                'src': one_state,
                'tgt': tgt,
                'neighbor_states': remaining_state,
                'neighbors_edge_state': (root_one_state,) + remaining_type,
            },
            ((root_state, one_state) + remaining_state, (root_one_state,) + remaining_type)
        ]
        for k in ips.deg_supp
        for remaining_state in product(ips.state_space, repeat=k - 1)
        for remaining_type in product(ips.edge_state_space, repeat=k - 1)
    ]


# TODO: merge this with jax_build_static_maps_vmap to reduce duplicate code
def jax_build_static_maps_vmap_dynamic_weights(ips, ode_state_space, vertex_state_space, state_to_index, ode_state_to_index,
                          gamma_logic_func, edge_state_space):
    """
    Pre-computes static structure for JAX.
    OPTIMIZED: Converts states to Integers IMMEDIATELY to prevent OOM errors.
    """

    """
        Pre-computes static structure for JAX. Returns arrays that describe:
        - Sparse matrix structure (rows, cols)
        - Which transitions are root vs neighbor jumps
        - Flattened arrays of ALL rate function arguments
        """

    rows = []
    cols = []
    root_jump_indices = []
    neighbor_jump_indices = []
    edge_jump_indices = []

    # Root jump : store as lists then convert to arrays
    root_src_list = []
    root_tgt_list = []
    root_neighbor_states_list = []
    root_neighbor_edge_states_list = []

    edge_src_list = []
    edge_tgt_list = []
    edge_neighbor_vertex_states_list = []

    # Gamma  structures
    gamma_dependency_indices = []
    gamma_weights = []

    # Flattened gamma term 
    gamma_src_list = []
    gamma_tgt_list = []
    gamma_neighbor_states_list = []
    gamma_neighbor_vertex_types_list = []
    gamma_neighbor_edge_types_list = []
    gamma_neighbor_edge_states_list = []

    # build mapping gamma gather map indices
    current_flat_index = 0
    gamma_gather_indices_list = []

    transition_idx = -1

    for src, tgt in product(vertex_state_space, repeat=2):
        # gather indices for vertex jumps
        if x_coordinate_apart(src, tgt):

            # neighbor_types here refers to the state of the dynamic edges
            for neighbor_types in edge_state_space:
                if len(neighbor_types) == len(src) - 1:
                    transition_idx += 1

                    row_idx = ode_state_to_index[(src, neighbor_types)]
                    col_idx = ode_state_to_index[(tgt, neighbor_types)]

                    rows.append(row_idx)
                    cols.append(col_idx)

                    # ROOT JUMP
                    if src[0] != tgt[0]:
                        root_jump_indices.append(transition_idx)
                        root_src_list.append(src[0])
                        root_tgt_list.append(tgt[0])
                        root_neighbor_states_list.append(src[1:])
                        root_neighbor_edge_states_list.append(neighbor_types)
                        
                    # NEIGHBOR JUMP
                    else:
                        neighbor_jump_indices.append(transition_idx)
                        changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

                        # Get gamma logic
                        needed_terms = gamma_logic_func(
                            ips, src[changed_index], 
                            tgt[changed_index],
                            src[changed_index], 
                            src[0],
                            root_one_state = neighbor_types[changed_index-1]
                        )

                        # Extract indices and weights
                        term_indices = [ode_state_to_index[s] for w, rate_args, s in needed_terms]
                        term_weights = [w for w, rate_args, s in needed_terms]

                        gamma_dependency_indices.append(term_indices)
                        gamma_weights.append(term_weights)

                        count = len(needed_terms)
                        indices = list(range(current_flat_index, current_flat_index + count))
                        gamma_gather_indices_list.append(indices)
                        current_flat_index += count

                        # Flatten all rate arguments for vectorized computation
                        for w, rate_args, s in needed_terms:
                            gamma_src_list.append(rate_args['src'])
                            gamma_tgt_list.append(rate_args['tgt'])
                            gamma_neighbor_states_list.append(rate_args['neighbor_states'])
                            gamma_neighbor_edge_states_list.append(rate_args['neighbors_edge_state'])
            
    # gather indices for edge jumps
    for src, tgt in product(edge_state_space, repeat=2):
        if x_coordinate_apart(src, tgt, x=2):
            # change indices must have 0
            changed_indices = [i for i in range(len(src)) if src[i] != tgt[i]]
            if changed_indices[0] != 0:
                continue
            else:
                # TODO: this can be optimized as edges are permutation invariant
                changed_index = changed_indices[1]
                for vertex_neighborhood_types in vertex_state_space:
                    transition_idx += 1

                    row_idx = ode_state_to_index[(vertex_neighborhood_types, src)]
                    col_idx = ode_state_to_index[(vertex_neighborhood_types, tgt)]

                    rows.append(row_idx)
                    cols.append(col_idx)
                    edge_jump_indices.append(transition_idx)

                    edge_src_list.append(src[changed_index])
                    edge_tgt_list.append(tgt[changed_index])
                    edge_neighbor_vertex_states_list.append((vertex_neighborhood_types[0], vertex_neighborhood_types[1]))


    # Convert lists to padded arrays
    # Root jumps
    max_neighbors = max(ips.deg_supp, default=0)
    num_root_jumps = len(root_src_list)

    neighbor_states_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(root_neighbor_states_list):
        neighbor_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    es_map = {s: i for i, s in enumerate(ips.edge_state_space)}
    root_neighbor_edge_states_padded = np.full((num_root_jumps, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(root_neighbor_edge_states_list):
        root_neighbor_edge_states_padded[i, :len(ns)] = [es_map[s] for s in ns]

    # Gamma terms
    num_gamma_terms = len(gamma_src_list)

    gamma_neighbor_states_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(gamma_neighbor_states_list):
        gamma_neighbor_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    gamma_neighbor_edge_states_padded = np.full((num_gamma_terms, max_neighbors), -1, dtype=np.int32)
    for i, ns in enumerate(gamma_neighbor_edge_states_list):
        gamma_neighbor_edge_states_padded[i, :len(ns)] = [es_map[s] for s in ns]

    # Pad gamma dependency indices and weights
    max_terms_per_jump = max((len(x) for x in gamma_dependency_indices), default=0)
    num_neighbor_jumps = len(neighbor_jump_indices)

    gamma_indices_padded = np.zeros((num_neighbor_jumps, max_terms_per_jump), dtype=np.int32)
    gamma_weights_padded = np.zeros((num_neighbor_jumps, max_terms_per_jump), dtype=np.float32)

    # Pad the Gather Map
    num_jumps = len(neighbor_jump_indices)
    max_terms_per_jump = max((len(x) for x in gamma_gather_indices_list), default=0)
    gamma_gather_map = np.zeros((num_jumps, max_terms_per_jump), dtype=np.int32)

    for i, inds in enumerate(gamma_gather_indices_list):
        gamma_gather_map[i, :len(inds)] = inds

    for i in range(num_neighbor_jumps):
        n_terms = len(gamma_dependency_indices[i])
        gamma_indices_padded[i, :n_terms] = gamma_dependency_indices[i]
        gamma_weights_padded[i, :n_terms] = gamma_weights[i]
    
    # pad edge jumps
    num_edge_jumps = len(edge_jump_indices)
    edge_neighbor_vertex_states_padded = np.full((num_edge_jumps, 2), -1, dtype=np.int32)
    for i, ns in enumerate(edge_neighbor_vertex_states_list):
        edge_neighbor_vertex_states_padded[i, :len(ns)] = [state_to_index[s] for s in ns]

    # Bundle everything
    static_args = {
        # Sparse matrix structure
        "rows": jnp.array(rows, dtype=jnp.int32),
        "cols": jnp.array(cols, dtype=jnp.int32),

        # Root jump 
        "root_idx_map": jnp.array(root_jump_indices, dtype=jnp.int32),
        "root_src": jnp.array([state_to_index[s] for s in root_src_list], dtype=jnp.int32),
        "root_tgt": jnp.array([state_to_index[s] for s in root_tgt_list], dtype=jnp.int32),
        "neighbors": jnp.array(neighbor_states_padded, dtype=jnp.int32),
        "neighbors_edge_states": jnp.array(root_neighbor_edge_states_padded, dtype=jnp.int32),

        # Neighbor jump 
        "neigh_idx_map": jnp.array(neighbor_jump_indices, dtype=jnp.int32),
        "gamma_indices": jnp.array(gamma_indices_padded, dtype=jnp.int32),
        "gamma_weights": jnp.array(gamma_weights_padded, dtype=jnp.float32),

        # edge jump 
        "edge_idx_map": jnp.array(edge_jump_indices, dtype=jnp.int32),
        "edge_src": jnp.array([es_map[s] for s in edge_src_list], dtype=jnp.int32),
        "edge_tgt": jnp.array([es_map[s] for s in edge_tgt_list], dtype=jnp.int32),
        "edge_neighbors": jnp.array(edge_neighbor_vertex_states_padded, dtype=jnp.int32),

        # Flattened gamma term  (for vectorized rate calls)
        "gamma_src": jnp.array([state_to_index[s] for s in gamma_src_list], dtype=jnp.int32),
        "gamma_tgt": jnp.array([state_to_index[s] for s in gamma_tgt_list], dtype=jnp.int32),
        "gamma_neighbors": jnp.array(gamma_neighbor_states_padded, dtype=jnp.int32),
        "gamma_neighbors_edge_states": jnp.array(gamma_neighbor_edge_states_padded, dtype=jnp.int32),

        # Mapping from neighbor jumps to gamma terms
        "gamma_gather_map": jnp.array(gamma_gather_map, dtype=jnp.int32),

        # Metadata
        "num_states": len(ode_state_space),
    }

    sparse_indices = jnp.stack([static_args["rows"], static_args["cols"]], axis=1)
    return static_args, sparse_indices
