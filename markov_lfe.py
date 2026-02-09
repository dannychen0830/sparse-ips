from sparseips.ips_class import ParticleSystem, MeanFieldParticleSystem
from sparseips.jump_ips_sim import simulate_mean_field_jump_process, get_particle_states_at_times
from sparseips.jax_mlfe import *

from jax.experimental import sparse
import jax.numpy as jnp
import diffrax
import lineax as lx
import optimistix as optx

import gc
from itertools import product
from collections import Counter


def sparse_diag(data: jnp.ndarray, size: int) -> sparse.BCOO:
    range_n = jnp.arange(size)
    indices = jnp.stack([range_n, range_n], axis=1)

    # --- Create BCOO Matrix ---
    return sparse.BCOO((data, indices), shape=(size, size))


def jax_mlfe_vector_field(t, p, args):
    """
    Calculates dp/dt entirely using vector operations.
    """

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


def jax_mlfe_vector_field_vmap(t, p, args):
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
        args["neighbors_edge_states"],
        p,
        t
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
        args["gamma_neighbors_edge_states"],
        p,
        t
    )

    # # Edge rate
    # edge_rate_caller = args["edge_rate_caller"]
    # edge_rates = edge_rate_caller(
    #     args["edge_src"],
    #     args["edge_tgt"],
    #     args["edge_neighbor_vertex_states"],
    #     p,
    #     t
    # )

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
    # all_rates = all_rates.at[args["edge_idx_map"]].set(edge_rates)

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


def simulate_markov_lfe(
        ips: ParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        vertex_type_init: dict[any, float] = None,
        edge_type_init: dict[any, float] = None,
        edge_state_init: dict[any, float] = None,
        num_grid_points: int = 100,
        solver_type: str = 'explicit',
        step_control: str = 'constant',
        vectorized: bool = None,
        verbose: bool = True,
        static_args: dict[str, jnp.ndarray] = None,
        sparse_indices: jnp.ndarray = None,
        rate_caller: callable = None
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[any]]]:
    # step up ode state space
    deg_dist = ips.deg_dist
    deg_supp = ips.deg_supp

    if ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        ode_state_space = vertex_state_space
    elif ips.vertex_type_space is not None and ips.edge_type_space is None and ips.edge_state_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        vertex_type_space = [(root,) + children for k in deg_supp for (root, children) in
                             product(ips.vertex_type_space, product(ips.vertex_type_space, repeat=k))]
        ode_state_space = [(state, type) for state in vertex_state_space for type in vertex_type_space if
                           len(state) == len(type)]
    elif ips.vertex_type_space is None and ips.edge_type_space is not None and ips.edge_state_space is None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        edge_type_space = [root_children for k in deg_supp for root_children in product(ips.edge_type_space, repeat=k)]
        ode_state_space = [(state, type) for state in vertex_state_space for type in edge_type_space if
                           len(state) == len(type) + 1]
    elif ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is not None:
        vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(ips.state_space, product(ips.state_space, repeat=k))]
        edge_state_space = [root_children for k in deg_supp for root_children in product(ips.edge_state_space, repeat=k)]
        ode_state_space = [(state, type) for state in vertex_state_space for type in edge_state_space if
                           len(state) == len(type) + 1]
    else:
        raise NotImplementedError(
            "JAX backend does not support both vertex and edge type spaces simultaneously."
        )

    index_to_ode_state_space = {i: state for i, state in enumerate(ode_state_space)}
    ode_state_space_to_index = {state: i for i, state in enumerate(ode_state_space)}

    # index state dependence
    if verbose:
        print('**** Building rate matrix structure ****')

    if vectorized is None:
        vectorized = hasattr(ips, 'rate_vectorized')
    if vectorized:
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

        if static_args is None or sparse_indices is None:
            # Build static structure
            static_args, sparse_indices = jax_build_static_maps_vmap(
                ips,
                ode_state_space,
                vertex_state_space, 
                ips.get_state_to_index_map(),
                ode_state_space_to_index,
                jax_gamma_index_builder_vmap,
                vertex_type_space=vertex_type_space if ips.vertex_type_space is not None else None,
                edge_type_space=edge_type_space if ips.edge_type_space is not None else None,
                edge_state_space=edge_state_space if ips.edge_state_space is not None else None
            )

                # Create rate caller
            if rate_caller is None:
                rate_caller = make_rate_caller(ips.rate_vectorized,
                                            rate_params,
                                            ips.vertex_type_space is not None,
                                            ips.edge_type_space is not None,
                                            ips.edge_state_space is not None)
       
        static_args["rate_caller"] = rate_caller

        # Create rate caller for edge rates 
        if ips.edge_state_space is not None:
            if not hasattr(ips, 'edge_rate_vectorized'):
                raise ValueError(
                    "JAX backend requires 'edge_rate_vectorized' method. "
                    "This method must use JAX-compatible operations (jnp.where, jax.vmap, etc.) "
                    "and accept vectorized inputs. See documentation for examples."
                )
            edge_rate_caller = make_edge_rate_caller(ips.edge_rate_vectorized, rate_params)
        else:
            edge_rate_caller = lambda *args: None
        static_args["edge_rate_caller"] = edge_rate_caller

        # define vector field
        term = diffrax.ODETerm(jax_mlfe_vector_field_vmap)
    else:
        if ips.global_interaction or ips.edge_state_space is not None:
            raise ValueError("Backend does not support global interactions in non-vectorized mode.")
        # Build static structure
        static_args, _ = jax_build_static_maps(ips, ode_state_space, vertex_state_space, ode_state_space_to_index,
                                               jax_gamma_index_builder,
                                               vertex_type_space=vertex_type_space if ips.vertex_type_space is not None else None,
                                               edge_type_space=edge_type_space if ips.edge_type_space is not None else None)
        # define vector field
        term = diffrax.ODETerm(jax_mlfe_vector_field)

    # Set up ODE solver
    # Initialize
    if ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is None:
        ode_init = [initial_conditions[state[0]] * deg_dist[len(state) - 1] * np.prod(
            [initial_conditions[child] for child in state[1:]]) for state in ode_state_space]
    elif ips.vertex_type_space is not None and ips.edge_type_space is None and ips.edge_state_space is None:
        ode_init = [
            initial_conditions[state[0]]
            * deg_dist[len(state) - 1]
            * np.prod([initial_conditions[child] for child in state[1:]])
            * np.prod([vertex_type_init[t] for t in type])
            for (state, type) in ode_state_space
        ]
    elif ips.vertex_type_space is None and ips.edge_type_space is not None and ips.edge_state_space is None:
        ode_init = [
            initial_conditions[state[0]]
            * deg_dist[len(state) - 1]
            * np.prod([initial_conditions[child] for child in state[1:]])
            * np.prod([edge_type_init[t] for t in type])
            for (state, type) in ode_state_space
        ]
    elif ips.vertex_type_space is None and ips.edge_type_space is None and ips.edge_state_space is not None:
        ode_init = [
            initial_conditions[state[0]]
            * deg_dist[len(state) - 1]
            * np.prod([initial_conditions[child] for child in state[1:]])
            * np.prod([edge_state_init[t] for t in type])
            for (state, type) in ode_state_space
        ]
    else:
        raise ValueError("Both vertex_type_space and edge_type_space cannot be set simultaneously.")
    y0 = jnp.array(ode_init)

    # Choose solver
    if solver_type == 'implicit':
        linear_solver = lx.GMRES(rtol=1e-2, atol=1e-2, restart=20)
        # linear_solver = lx.AutoLinearSolver(well_posed=False)
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
        step_controller = diffrax.PIDController(rtol=1e-9, atol=1e-12)
    elif step_control == 'constant':
        step_controller = diffrax.ConstantStepSize()
    else:
        raise ValueError(f'Unknown step control type: {step_control}')

    if verbose:
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
        progress_meter=diffrax.TqdmProgressMeter() if verbose else diffrax.NoProgressMeter(),
    )

    return sol.ts, sol.ys.transpose(), index_to_ode_state_space


# Deprecated
# def simulate_markov_lfe_mf(
#         ips: ParticleSystem,
#         initial_conditions: dict[any, float],
#         max_time: float,
#         num_particles: int = 500,
#         seed: int = 42,
#         num_grid_points: int = 100,
#         gamma: callable = None
# ) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[any]]]:
#     # model parameters
#     deg_dist = ips.get_empirical_degree_distribution()
#     deg_supp = [i for (i, p) in deg_dist.items() if p > 0]

#     # track all possible root-children marginals
#     vertex_state_space = [(root,) + children for k in deg_supp for (root, children) in
#                           product(ips.state_space, product(ips.state_space, repeat=k))]
#     ode_state_space = [(root,) + children for k in deg_supp for (root, children) in
#                        product(ips.state_space, product(ips.state_space, repeat=k))]

#     index_to_ode_state_space = {i: state for i, state in enumerate(ode_state_space)}
#     ode_state_space_to_index = {state: i for i, state in index_to_ode_state_space.items()}

#     if gamma is None:
#         # define the Markov local-field jump rate
#         def gamma(src: tuple, tgt: tuple, root_state, one_state, marginal_prob: dict[tuple, float]) -> float:
#             numerator = sum(
#                 (2 + len(remaining_state)) *
#                 marginal_prob.get((root_state, one_state) + remaining_state, 0) * ips.rate(src, tgt, (
#                     one_state,) + remaining_state)
#                 for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
#             )
#             denominator = sum(
#                 (2 + len(remaining_state)) * marginal_prob.get((root_state, one_state) + remaining_state, 0)
#                 for k in deg_supp for remaining_state in product(ips.state_space, repeat=k - 1)
#             )
#             return numerator / denominator if denominator > 0 else 0

#     class MLFEParticleSystem(MeanFieldParticleSystem):
#         def __init__(self, ode_state_space: list[any], num_particles: int, name: str = None):
#             super().__init__(state_space=ode_state_space, num_particles=num_particles, name=name)

#         def rate(self, src: any,
#                  tgt: any,
#                  meas: dict[tuple[any], float]) -> float:
#             if one_coordinate_apart(src, tgt):
#                 # find the index of the changed coordinate
#                 changed_index = next(i for i in range(len(src)) if src[i] != tgt[i])

#                 # if the root jumped, return usual rate
#                 if changed_index == 0:
#                     return ips.rate(src[0], tgt[0], src[1:])
#                 # otherwise, take conditional rate
#                 else:
#                     return gamma(src[changed_index], tgt[changed_index], src[changed_index], src[0], meas)
#             return 0.0

#     # calculate initial conditions on ode_state_space given i.i.d. initial conditions on vertices
#     mf_init = {}
#     for i in range(num_particles):
#         # pick offspring number from degree distribution
#         k = np.random.choice(deg_supp, p=[deg_dist[i] for i in deg_supp])
#         # pick states for root and k leafs
#         init_state = tuple(
#             [np.random.choice(ips.state_space, p=[initial_conditions[s] for s in ips.state_space]) for _ in
#              range(k + 1)])
#         # add to initial conditions
#         mf_init[i] = init_state

#     # create the mean-field particle system
#     mfps = MLFEParticleSystem(ode_state_space, num_particles, name=f"Mean-field approximation for MLFE for {ips.name}")
#     # simulate the mean-field particle system (timed)
#     mf_jump_list = simulate_mean_field_jump_process(mfps=mfps, initial_conditions=mf_init, max_time=max_time, seed=seed)

#     # convert jump list to array with time and state
#     time_points = np.linspace(0, max_time, num_grid_points)
#     time_state_dict = get_particle_states_at_times(mf_jump_list, mf_init, list(time_points))

#     time_state_ndarray = np.zeros((len(ode_state_space), num_grid_points))
#     for t_idx, state_at_time in enumerate(time_state_dict):
#         counter = Counter(state_at_time.values())
#         for state in ode_state_space:
#             time_state_ndarray[ode_state_space_to_index[state], t_idx] = counter[state] / num_particles

#     return time_points, time_state_ndarray, index_to_ode_state_space
