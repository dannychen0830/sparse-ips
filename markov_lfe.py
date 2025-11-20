import numpy as np
from sparseips.ips_class import ParticleSystem, MeanFieldParticleSystem
from sparseips.jump_ips_sim import simulate_mean_field_jump_process, get_particle_states_at_times
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, diags

import jax.debug
from jax.experimental import sparse
import jax.numpy as jnp
import diffrax
import lineax as lx
import optimistix as optx

from itertools import product
from collections import Counter
from datetime import datetime


# TODO: fix reduced mlfe
# def jump_rate_for_red_mlfe(
#         state_space: List[Any],
#         state_to_index: Dict[Any, int],
#         b: Dict[Tuple[Any, Any], float],
#         rho: Dict[Tuple[Any, Any], float]
# ) -> Callable:
#     # define the rate function to be returned
#     def rate(src_state, tgt_state, neighbors_state, **kwargs):
#         # check if src_state is the zero state
#         if state_to_index[src_state] == 0:
#             return sum(b[(a, tgt_state)] * sum(1 for state in neighbors_state if state == a) for a in state_space)
#         # if not, transition is independent of neighbor states
#         else:
#             return rho[(src_state, tgt_state)]
#
#     return rate
#
#
# def simulate_reduced_markov_lfe(
#         state_space: List[Any],
#         state_to_index: Dict[Any, int],
#         b: Dict[Tuple[Any, Any], float],
#         rho: Dict[Tuple[Any, Any], float],
#         deg_dist: List[float],
#         phi: Callable[[float], float],
#         max_time: float,
#         initial_conditions: Dict[Any, float],
#         num_grid_points: int = 100
# ):
#     # define simulation parameters
#     num_states = len(state_space)
#     deg_supp = [i for (i,p) in enumerate(deg_dist) if p > 0]
#     deg_dist_no_zero = np.array([p for p in deg_dist if p > 0])
#     initial_conditions_array = np.array([v for v in initial_conditions.values()])
#
#     # initial condition for f_0, f_a for a = {1, ..., m}
#     f = initial_conditions_array
#     # initial condition for F
#     F = np.array([0.0])
#     # initial condition for P_0,0;k = 1 for k such that theta(k) > 0
#     # initial condition for P_0,a;k = 0 for a > 0 and k such that theta(k) > 0 is 0.0
#     p0ak = np.zeros(shape=(num_states, len(deg_supp)))
#     p0ak[0, :] = 1.0
#     # initial condition for P_a,c
#     pac = np.eye(num_states - 1)
#
#     ode_initial_conditions = np.concatenate((f, F, p0ak.flatten(), pac.flatten()))
#
#     # convert beta and rho from dictionary to numpy array
#     b_mat = np.zeros(shape=(num_states, num_states))
#     rho_mat = np.zeros(shape=(num_states, num_states))
#     for i in state_space:
#         for j in state_space:
#             try:
#                 b_mat[state_to_index[i], state_to_index[j]] = b[(i,j)]
#             except KeyError:
#                 b_mat[state_to_index[i], state_to_index[j]] = 0.0
#             try:
#                 rho_mat[state_to_index[i], state_to_index[j]] = rho[(i,j)]
#             except KeyError:
#                 rho_mat[state_to_index[i], state_to_index[j]] = 0.0
#     # diagonal of rho is negative row sum
#     for i in range(num_states):
#         rho_mat[i,i] = -np.sum(rho_mat[i,:])
#     b_vec = np.sum(b_mat, axis=1)
#     b_mat_red = b_mat[1:, 1:]
#     rho_mat_red = rho_mat[1:, 1:]
#     k = np.array(deg_supp)
#
#     def red_mlfe_ode(t, y):
#         f = y[:num_states]
#         F = y[num_states:num_states + 1]
#         p0ak = y[num_states + 1:num_states + 1 + len(deg_supp) * num_states]
#         pac = y[num_states + 1 + len(deg_supp) * num_states:]
#
#         f = f.reshape((num_states,))
#         F = F.reshape((1,))
#         p0ak = p0ak.reshape((num_states, len(deg_supp)))
#         pac = pac.reshape((num_states - 1, num_states - 1))
#
#         # compute the derivatives
#         df0 = np.dot(b_vec, f) * f[0] * (1 - phi(F[0]))
#         dfa = f[0] * phi(F) * b_mat_red.T @ f[1:] + rho_mat_red.T @ f[1:] + f[1:] * np.dot(b_vec, f) - f[1:] * b_vec[1:]
#         dF = np.dot(b_vec, f)
#
#         dp0ak = np.zeros(shape=(num_states, len(deg_supp)))
#         dp0ak[0,:] = -np.dot(b_vec, f) * k * p0ak[0,:]
#         dp0ak[1:,:] = (p0ak[0,:] * k * b_mat_red.T @ f[1:] + (p0ak[1:,:].T @ rho_mat_red).squeeze()).reshape((num_states - 1, len(deg_supp)))
#
#         dpac = pac @ rho_mat_red
#
#         return np.concatenate((df0.reshape((1,)), dfa, dF.reshape((1,)), dp0ak.flatten(), dpac.flatten()))
#
#     # solve the ode
#     t_span = (0, max_time)
#     t_eval = np.linspace(0, max_time, num_grid_points)
#     sol = solve_ivp(red_mlfe_ode, t_span, ode_initial_conditions, t_eval=t_eval, method='RK45')
#
#     # extract the solution
#     def convert_to_prob(y):
#         f = y[:num_states]
#         F = y[num_states:num_states + 1]
#         p0ak = y[num_states + 1:num_states + 1 + len(deg_supp) * num_states]
#         pac = y[num_states + 1 + len(deg_supp) * num_states:]
#
#         f = f.reshape((num_states,))
#         F = F.reshape((1,))
#         p0ak = p0ak.reshape((num_states, len(deg_supp)))
#         pac = pac.reshape((num_states - 1, num_states - 1))
#
#         return initial_conditions_array[0] * deg_dist_no_zero @ p0ak[1:,:].T + pac.T @ initial_conditions_array[1:]
#
#     # apply convert_to_prob to the solution
#     prob_sol = np.array([convert_to_prob(y) for y in sol.y.T])
#
#     return t_eval, prob_sol, sol


def one_coordinate_apart(tuple1: tuple, tuple2: tuple) -> bool:
    """
    Check if two tuples are one coordinate apart, i.e., they differ in exactly one coordinate.
    :param tuple1: first input tuple
    :param tuple2: second input tuple
    :return: True if the tuples differ in exactly one coordinate, False otherwise
    """
    return len(tuple1) == len(tuple2) and sum(x != y for x, y in zip(tuple1, tuple2)) == 1


def compute_gamma(ips, deg_supp):
    if ips.vertex_type_space is None and ips.edge_type_space is None:
        def gamma(src: tuple, tgt: tuple, root_state, one_state, marginal_prob: dict[tuple, float],
                  **kwargs) -> float:
            numerator = sum(
                (1 + len(remaining_state)) *
                marginal_prob[(root_state, one_state) + remaining_state] *
                ips.rate(src, tgt, (one_state,) + remaining_state,
                         meas=marginal_prob if ips.global_interaction else None)
                for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
            )
            denominator = sum(
                (1 + len(remaining_state)) * marginal_prob[(root_state, one_state) + remaining_state]
                for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
            )
            return numerator / denominator if denominator > 0 else 0

    elif ips.vertex_type_space is not None and ips.edge_type_space is None:
        def gamma(src: tuple, tgt: tuple, root_state, one_state, marginal_prob: dict[tuple, float], root_type,
                  one_type) -> float:
            numerator = sum(
                (2 + len(remaining_state)) *
                marginal_prob[((root_state, one_state) + remaining_state, (root_type, one_type) + remaining_type)]
                * ips.rate(src, tgt, (one_state,) + remaining_state,
                           neighbors_vertex_type=(root_type, one_type) + remaining_type,
                           meas=marginal_prob if ips.global_interaction else None)
                for k in deg_supp
                for remaining_state in product(ips.state_space, repeat=k - 1)
                for remaining_type in product(ips.vertex_type_space, repeat=k - 1)
            )
            denominator = sum(
                (2 + len(remaining_state)) * marginal_prob[
                    ((root_state, one_state) + remaining_state, (root_type, one_type) + remaining_type)]
                for k in deg_supp
                for remaining_state in product(ips.state_space, repeat=k - 1)
                for remaining_type in product(ips.vertex_type_space, repeat=k - 1)
            )
            return numerator / denominator if denominator > 0 else 0

    elif ips.vertex_type_space is None and ips.edge_type_space is not None:
        def gamma(src: tuple, tgt: tuple, root_state, one_state, marginal_prob: dict[tuple, float],
                  root_one_type) -> float:
            numerator = sum(
                (1 + len(remaining_state)) *
                marginal_prob[((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)]
                * ips.rate(src, tgt, (one_state,) + remaining_state,
                           neighbors_edge_type=(root_one_type,) + remaining_type,
                           meas=marginal_prob if ips.global_interaction else None)
                for k in deg_supp
                for remaining_state in product(ips.state_space, repeat=k - 1)
                for remaining_type in product(ips.edge_type_space, repeat=k - 1)
            )
            denominator = sum(
                (1 + len(remaining_state)) * marginal_prob[
                    ((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)]
                for k in deg_supp
                for remaining_state in product(ips.state_space, repeat=k - 1)
                for remaining_type in product(ips.edge_type_space, repeat=k - 1)
            )
            return numerator / denominator if denominator > 0 else 0

    return gamma


def sparse_diag(data: jnp.ndarray, size: int) -> sparse.BCOO:
    range_n = jnp.arange(size)
    indices = jnp.stack([range_n, range_n], axis=1)

    # --- Create BCOO Matrix ---
    return sparse.BCOO((data, indices), shape=(size, size))


def gamma_logic_func_builder(ips, src, tgt, root_state, one_state, root_type=None, one_type=None, root_one_type=None):
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


def build_static_maps(ips, ode_state_space, vertex_state_space, ode_state_to_index, gamma_logic_func,
                      vertex_type_space=None, edge_type_space=None):
    """
    Analyzes the graph structure and returns static arrays for JAX.
    """
    rows = []
    cols = []

    # Lists to store data needed for Root Jumps vs Neighbor Jumps
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

    # Data for Root Calculation
    root_rate_params = []  # Store static args for ips.rate if needed

    # 1. Iterate ALL transitions (Python Loop - Runs ONCE)
    transition_idx = -1
    for src, tgt in product(vertex_state_space, repeat=2):
        is_valid_transition = one_coordinate_apart(src, tgt)

        if is_valid_transition:
            # TODO: a lot of duplicate code here... bad!!!
            # determine jump type
            if ips.vertex_type_space is None and ips.edge_type_space is None:
                transition_idx += 1
                row_idx = ode_state_to_index[src]
                col_idx = ode_state_to_index[tgt]
                rows.append(row_idx)
                cols.append(col_idx)

                # root jump
                if src[0] != tgt[0]:
                    root_jump_indices.append(transition_idx)
                    root_rates.append(ips.rate(src[0], tgt[0], src[1:]))
                # leaf jump
                else:
                    # NEIGHBOR JUMP (Uses Gamma)
                    neighbor_jump_indices.append(transition_idx)

                    # compute tuples (weight, rate, state) needed for gamma calculation
                    changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])
                    needed_terms = gamma_logic_func(ips, src[changed_index], tgt[changed_index], src[0], tgt[0])

                    term_indices = [ode_state_to_index[s] for w, r, s in needed_terms]
                    term_rates = [r for w, r, s in needed_terms]
                    term_weights = [w for w, r, s in needed_terms]

                    gamma_dependency_indices.append(term_indices)
                    gamma_weights.append(term_weights)
                    gamma_rates.append(term_rates)
            elif ips.vertex_type_space is not None or ips.edge_type_space is None:
                for neighbor_types in vertex_type_space:
                    if len(neighbor_types) == len(src):

                        transition_idx += 1
                        row_idx = ode_state_to_index[(src, neighbor_types)]
                        col_idx = ode_state_to_index[(tgt, neighbor_types)]
                        rows.append(row_idx)
                        cols.append(col_idx)

                        if src[0] != tgt[0]:
                            root_jump_indices.append(transition_idx)
                            root_rates.append(ips.rate(src[0], tgt[0], src[1:],
                                                       neighbors_vertex_type=neighbor_types))
                        # leaf jump
                        else:
                            # NEIGHBOR JUMP (Uses Gamma)
                            neighbor_jump_indices.append(transition_idx)

                            # compute tuples (weight, rate, state) needed for gamma calculation
                            changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])
                            needed_terms = gamma_logic_func(ips, src[changed_index], tgt[changed_index], src[0], tgt[0],
                                                            root_type=neighbor_types[changed_index],
                                                            one_type=neighbor_types[0])

                            term_indices = [ode_state_to_index[s] for w, r, s in needed_terms]
                            term_rates = [r for w, r, s in needed_terms]
                            term_weights = [w for w, r, s in needed_terms]

                            gamma_dependency_indices.append(term_indices)
                            gamma_weights.append(term_weights)
                            gamma_rates.append(term_rates)

            elif ips.vertex_type_space is None and ips.edge_type_space is not None:
                for neighbor_types in edge_type_space:
                    if len(neighbor_types) == len(src) - 1:

                        transition_idx += 1
                        row_idx = ode_state_to_index[(src, neighbor_types)]
                        col_idx = ode_state_to_index[(tgt, neighbor_types)]
                        rows.append(row_idx)
                        cols.append(col_idx)

                        if src[0] != tgt[0]:
                            root_jump_indices.append(transition_idx)
                            root_rates.append(ips.rate(src[0], tgt[0], src[1:], neighbors_edge_type=neighbor_types))
                        # leaf jump
                        else:
                            # NEIGHBOR JUMP (Uses Gamma)
                            neighbor_jump_indices.append(transition_idx)

                            # compute tuples (weight, rate, state) needed for gamma calculation
                            changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])
                            needed_terms = gamma_logic_func(ips, src[changed_index], tgt[changed_index], src[0], tgt[0],
                                                            root_one_type=neighbor_types[changed_index - 1])

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


def print_progress(t):
    """Standard Python function to execute the print action."""
    print(f"**** Time: {t}")


def mlfe_vector_field(t, p, args):
    """
    Calculates dp/dt entirely using vector operations.
    """
    jax.debug.callback(print_progress, t)

    # Unpack static args
    gamma_indices = args["gamma_indices"]  # Shape: (Num_Neigh_Jumps, Max_Terms)
    gamma_weights = args["gamma_weights"]  # Shape: (Num_Neigh_Jumps, Max_Terms)
    gamma_rates = args["gamma_rates"]  # Shape: (Num_Neigh_Jumps, Max_Terms)

    # -------------------------------------------------
    # 1. Calculate Gamma Rates (Vectorized)
    # -------------------------------------------------
    # Gather probabilities for all terms in all gamma sums simultaneously
    # Shape: (Num_Neigh_Jumps, Max_Terms)
    term_probs = p[gamma_indices]

    # Calculate Denominators (Sum of weight * prob)
    denom_terms = term_probs * gamma_weights
    denoms = jnp.sum(denom_terms, axis=1)

    # Calculate Numerators
    # If ips.rate is constant, it's already in weights.
    # If ips.rate depends on 'p', you calculate it here similarly using vector ops.
    num_terms = denom_terms * gamma_rates
    nums = jnp.sum(num_terms, axis=1)

    # Avoid division by zero
    neigh_rates = nums / (denoms + 1e-12)

    # -------------------------------------------------
    # 2. Calculate Root Rates (Vectorized)
    # -------------------------------------------------
    # Assuming a simple rate function for root jumps here.
    # You would use similar gathering logic if it depends on p.
    root_rates = args["root_rates"]  # Placeholder

    # -------------------------------------------------
    # 3. Combine Rates and Update Matrix
    # -------------------------------------------------
    # We need to put rates back into the order of 'rows' and 'cols'
    total_transitions = len(args["rows"])
    all_rates = jnp.zeros(total_transitions)

    # Scatter the calculated rates into the full rates array
    all_rates = all_rates.at[args["neigh_idx_map"]].set(neigh_rates)
    all_rates = all_rates.at[args["root_idx_map"]].set(root_rates)

    # Create BCOO Matrix (O(1) inside JIT)
    sparse_indices = jnp.stack([args["rows"], args["cols"]], axis=1)
    Q_off = sparse.BCOO((all_rates, sparse_indices), shape=(args["num_states"], args["num_states"]))

    # Diagonal Adjustment
    row_sums = sparse.bcoo_reduce_sum(Q_off, axes=[1]).todense().reshape(-1)
    Q_final = Q_off - sparse_diag(row_sums, args["num_states"])

    return Q_final.T @ p


def simulate_markov_lfe(
        ips: ParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        vertex_type_init: dict[any, float] = None,
        edge_type_init: dict[any, float] = None,
        num_grid_points: int = 100,
        solver_type: str = 'explicit',
        step_control: str = 'constant',
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[any]]]:
    # model parameters
    deg_dist = ips.get_empirical_degree_distribution()
    deg_supp = [i for (i, p) in deg_dist.items() if p > 0]

    # track all possible root-children marginals
    if ips.vertex_type_space is None and ips.edge_type_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        ode_state_space = [(root,) + children for k in deg_supp for (root, children) in
                           product(ips.state_space, product(ips.state_space, repeat=k))]

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
        edge_type_space = [children for k in deg_supp for (root, children) in
                           product(ips.edge_type_space, product(ips.edge_type_space, repeat=k))]
        ode_state_space = [(state, type) for state in vertex_state_space for type in edge_type_space if
                           len(state) == len(type) + 1]

    index_to_ode_state_space = {i: state for i, state in enumerate(ode_state_space)}
    ode_state_space_to_index = {state: i for i, state in enumerate(ode_state_space)}

    # build args for JIT-compiled rate matrix function
    print('**** Building rate matrix structure ****')
    static_args, _ = build_static_maps(ips, ode_state_space, vertex_state_space, ode_state_space_to_index,
                                       gamma_logic_func_builder,
                                       vertex_type_space=vertex_type_space if ips.vertex_type_space is not None else None,
                                       edge_type_space=edge_type_space if ips.edge_type_space is not None else None)

    # calculate initial conditions on ode_state_space given i.i.d. initial conditions on vertices
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
            * np.prod([edge_type_init[t] for t in type]) / 2
            for (state, type) in ode_state_space
        ]

    y0 = jnp.array(ode_init)

    term = diffrax.ODETerm(mlfe_vector_field)

    if solver_type == 'implicit':
        linear_solver = lx.GMRES(
            rtol=1e-3,
            atol=1e-3,
            restart=20  # Number of iterations before restarting (tuning parameter)
        )

        root_finder = optx.Newton(
            rtol=1e-3,
            atol=1e-3,
            linear_solver=linear_solver  # Inject GMRES here
        )

        solver = diffrax.Kvaerno3(root_finder=root_finder)
    elif solver_type == 'explicit':
        solver = diffrax.Dopri5()
    else:
        raise ValueError(f'Unknown solver type: {solver_type}')

    # Define output times
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, max_time, num_grid_points))

    # PID Controller is CRITICAL for stiff systems
    if step_control == 'adative':
        step_controller = diffrax.PDController(rtol=1e-3, atol=1e-3)
    elif step_control == 'constant':
        step_controller = diffrax.ConstantStepSize()
    else:
        raise ValueError(f'Unknown step control type: {step_control}')

    # 4. RUN
    print('**** Running simulation ****')
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=max_time,
        dt0=0.01,  # Initial step size guess (scalar!)
        y0=y0,
        args=static_args,
        stepsize_controller=step_controller,
        saveat=saveat,
        max_steps=100000  # Safety limit
    )

    return sol.ts, sol.ys.transpose(), index_to_ode_state_space


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
