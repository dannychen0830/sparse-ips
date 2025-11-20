import numpy as np
from sparseips.ips_class import ParticleSystem, MeanFieldParticleSystem
from sparseips.jump_ips_sim import simulate_mean_field_jump_process, get_particle_states_at_times
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, diags

from jax.experimental import sparse
import jax.numpy as jnp

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
                (2 + len(remaining_state)) *
                marginal_prob[((root_state, one_state) + remaining_state, (root_one_type,) + remaining_type)]
                * ips.rate(src, tgt, (one_state,) + remaining_state,
                           neighbors_edge_type=(root_one_type,) + remaining_type,
                           meas=marginal_prob if ips.global_interaction else None)
                for k in deg_supp
                for remaining_state in product(ips.state_space, repeat=k - 1)
                for remaining_type in product(ips.edge_type_space, repeat=k - 1)
            )
            denominator = sum(
                (2 + len(remaining_state)) * marginal_prob[
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


def simulate_markov_lfe(
        ips: ParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        vertex_type_init: dict[any, float] = None,
        edge_type_init: dict[any, float] = None,
        num_grid_points: int = 100,
        gamma: callable = None,
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

    gamma = compute_gamma(ips, deg_supp)

    # identify non-zero transitions rates
    # TODO: further reduce allowable transitions
    rows = []
    cols = []

    for src, tgt in product(vertex_state_space, repeat=2):
        if one_coordinate_apart(src, tgt):
            if ips.vertex_type_space is None and ips.edge_type_space is None:
                rows.append(ode_state_space_to_index[src])
                cols.append(ode_state_space_to_index[tgt])
            elif ips.vertex_type_space is not None and ips.edge_type_space is None:
                for neighbors_types in vertex_type_space:
                    if len(src) == len(neighbors_types):
                        rows.append(ode_state_space_to_index[(src, neighbors_types)])
                        cols.append(ode_state_space_to_index[(tgt, neighbors_types)])
            elif ips.vertex_type_space is None and ips.edge_type_space is not None:
                for neighbors_types in edge_type_space:
                    if len(src) == len(neighbors_types) - 1:
                        rows.append(ode_state_space_to_index[(src, neighbors_types)])
                        cols.append(ode_state_space_to_index[(tgt, neighbors_types)])

    sparse_indices = jnp.array([[rows[i], cols[i]] for i in range(len(rows))])

    def mlfe_ode(t, p):
        # convert p to dictionary from ode_state_space to probabilities
        marginal_prob = {ode_state_space[i]: p[i] for i in range(len(ode_state_space))}

        # extract src, tgt relationships in the expanded ode_state_space
        ode_rate = []
        for row_idx, col_idx in zip(rows, cols):
            src = index_to_ode_state_space[row_idx]
            tgt = index_to_ode_state_space[col_idx]

            if ips.vertex_type_space is None and ips.edge_type_space is None:
                # find the index of the changed coordinate
                changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

                # if the root jumped, return usual rate
                if changed_index == 0:
                    ode_rate.append(
                        ips.rate(src[0], tgt[0], src[1:], meas=marginal_prob if ips.global_interaction else None))
                # otherwise, take conditional rate
                else:
                    ode_rate.append(
                        gamma(src[changed_index], tgt[changed_index], src[changed_index], src[0], marginal_prob))

            elif ips.vertex_type_space is not None and ips.edge_type_space is None:
                changed_index = next(i for i in range(len(src[0])) if src[0][i] != tgt[0][i])
                if changed_index == 0:
                    ode_rate.append(
                        ips.rate(src[0][0],
                                 tgt[0][0],
                                 src[0][1:],
                                 neighbors_vertex_type=src[1],
                                 meas=marginal_prob if ips.global_interaction else None)
                    )
                else:
                    ode_rate.append(
                        gamma(src[0][changed_index],
                              tgt[0][changed_index],
                              src[0][changed_index],
                              src[0][0],
                              marginal_prob,
                              root_type=src[1][changed_index],
                              one_type=src[1][0]
                              )
                    )
            elif ips.vertex_type_space is None and ips.edge_type_space is not None:
                changed_index = next(i for i in range(len(src[0])) if src[0][i] != tgt[0][i])

                if changed_index == 0:
                    ode_rate.append(
                        ips.rate(src[0][changed_index],
                                 tgt[0][changed_index],
                                 src[0][1:],
                                 neighbors_edge_type=src[1],
                                 meas=marginal_prob if ips.global_interaction else None)
                    )
                else:
                    ode_rate.append(gamma(src[0][changed_index],
                                          tgt[0][changed_index],
                                          src[0][changed_index],
                                          src[0][0],
                                          marginal_prob,
                                          root_one_type=src[1][changed_index - 1]))

        sparse_Q_off_diag = sparse.BCOO((jnp.array(ode_rate), sparse_indices), shape=(len(ode_state_space), len(ode_state_space)))

        row_sums = sparse.bcoo_reduce_sum(sparse_Q_off_diag, axes=[1]).todense().reshape(-1)
        sparse_Q = sparse_Q_off_diag - sparse_diag(row_sums, len(ode_state_space))

        # print time
        print(f"t = {t}")
        return sparse_Q.transpose() @ p

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

    # solve the ode
    t_span = (0, max_time)
    t_eval = np.linspace(0, max_time, num_grid_points)
    sol = solve_ivp(mlfe_ode, t_span, ode_init, t_eval=t_eval, method='RK23')

    return t_eval, sol.y, index_to_ode_state_space


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
