from ips_class import MeanFieldParticleSystem

import numpy as np
from scipy.integrate import solve_ivp
import casadi as ca


def simulate_mean_field(
        mfps: MeanFieldParticleSystem,
        initial_conditions: dict[any, float],
        max_time: float,
        num_grid_points: int = 100,
):
    d = len(mfps.state_space)

    # simulate mean-field ode
    def mf_ode(t, y):
        # set up rate matrix
        rate_matrix = np.zeros((d, d))
        for i, src in enumerate(mfps.state_space):
            for j, tgt in enumerate(mfps.state_space):
                rate_matrix[i, j] = mfps.rate(src, tgt, y)

        # compute derivative
        dydt = np.zeros(d)
        for i in range(d):
            inflow = sum(rate_matrix[j, i] * y[j] for j in range(d) if j != i)
            outflow = sum(rate_matrix[i, j] * y[i] for j in range(d) if j != i)
            dydt[i] = inflow - outflow

        return dydt

    # initial condition vector
    y0 = np.array([initial_conditions.get(state, 0.0) for state in mfps.state_space])
    t_span = (0, max_time)
    t_eval = np.linspace(0, max_time, num_grid_points)
    sol = solve_ivp(mf_ode, t_span, y0, t_eval=t_eval, vectorized=True)

    return sol.t, sol.y


def solve_mf_optimal_control(

):
    pass


### testing area
import matplotlib.pyplot as plt


# ============================================================================
# High-Level API for Kolmogorov Optimal Control
# ============================================================================

def solve_kolmogorov_optimal_control(
        rate_function,
        running_cost,
        terminal_cost,
        n_states,
        n_controls,
        p_initial,
        T,
        N=50,
        control_bounds=(-1.0, 1.0),
        integration_method='rk4',
        solver_options=None,
        initial_guess=None,
        verbose=True
):
    """
    Solve optimal control problem for forward Kolmogorov equation.

    Parameters
    ----------
    rate_function : callable
        Function signature: rate_function(i, j, p, u, n_states) -> scalar
        Returns transition rate from state i to state j.

    running_cost : callable
        Function signature: running_cost(p, u) -> scalar
        Returns instantaneous cost at state p with control u.

    terminal_cost : callable
        Function signature: terminal_cost(p) -> scalar
        Returns terminal cost for final state p.

    n_states : int
        Number of states in the Markov chain.

    n_controls : int
        Dimension of control vector.

    p_initial : array_like
        Initial probability distribution (must sum to 1).

    T : float
        Time horizon.

    N : int, optional
        Number of time steps (default: 50).

    control_bounds : tuple, optional
        (lower, upper) bounds on control (default: (-1, 1)).

    integration_method : str, optional
        'euler', 'rk4', or 'cvodes' (default: 'rk4').

    solver_options : dict, optional
        Options passed to IPOPT solver.

    initial_guess : dict, optional
        Dictionary with keys 'P' and 'U' for warm-starting.

    verbose : bool, optional
        Print solver output (default: True).

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'P': optimal state trajectory (n_states × N+1)
        - 'U': optimal control trajectory (n_controls × N)
        - 't': time vector
        - 'cost': optimal cost value
        - 'success': whether solver converged
        - 'opti': the Opti object (for advanced users)
    """

    def create_sparse_mx(rows, cols, values, n_rows, n_cols):
        """
        Robustly creates a sparse MX matrix from triplets.

        Parameters
        ----------
        rows : list of int
        cols : list of int
        values : list of MX (symbolic expressions)
        n_rows : int
        n_cols : int
        """

        # 1. Create a list of indices [0, 1, 2, ..., N-1]
        # These point to the original locations in your 'values' list
        indices = list(range(len(values)))

        # 2. Create a 'Mapping Matrix' using DM (which handles sorting/summing)
        # We put the INDICES into the matrix, not the values.
        # DM.triplet automatically sums duplicates. If you have multiple entries
        # for (i,j), this logic might need a tweak, but for unique (i,j) it's perfect.
        mapping = ca.DM.triplet(rows, cols, indices, n_rows, n_cols)

        # 3. Extract the sparsity pattern (CasADi has now determined the structure)
        sparsity = mapping.sparsity()

        # 4. Extract the reordered indices
        # mapping.nonzeros() returns the values (which are our indices)
        # in the order CasADi stores them (Column-Major).
        permuted_indices = mapping.nonzeros()

        # 5. Reorder your symbolic values list to match CasADi's expectation
        # We need to cast permuted_indices to int because they come out as floats from DM
        sorted_values = [values[int(idx)] for idx in permuted_indices]

        # 6. Create the final MX
        # Now 'sorted_values' aligns perfectly with 'sparsity'
        return ca.MX(sparsity, ca.vertcat(*sorted_values))

    # Build transition rate matrix from rate function
    def build_rate_matrix(p, u):
        rows = []
        cols = []
        values = []

        for i in range(n_states - 1):
            j = i + 1
            rows.append(i)
            cols.append(j)
            values.append(rate_function(i, j, p, u, n_states))

        for i in range(1, n_states):
            j = i - 1
            rows.append(i)
            cols.append(j)
            values.append(rate_function(i, j, p, u, n_states))

        Q_off = create_sparse_mx(rows, cols, values, n_states, n_states)

        row_sums = ca.sum2(Q_off)
        Q = Q_off - ca.diag(row_sums)

        return Q

    # Forward Kolmogorov dynamics
    def dynamics(p, u):
        Q = build_rate_matrix(p, u)
        return ca.mtimes(Q.T, p)

    # Create integrator based on method
    dt = T / N

    # def integrate_step(p, u):
    #     if integration_method == 'euler':
    #         return p + dt * dynamics(p, u)
    #
    #     elif integration_method == 'rk4':
    #         k1 = dynamics(p, u)
    #         k2 = dynamics(p + dt / 2 * k1, u)
    #         k3 = dynamics(p + dt / 2 * k2, u)
    #         k4 = dynamics(p + dt * k3, u)
    #         return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #
    #     elif integration_method == 'cvodes':
    #         # Use CasADi's built-in integrator
    #         ode = {'x': p, 'p': u, 'ode': dynamics(p, u)}
    #         F = ca.integrator('F', 'cvodes', ode, {'t0': 0, 'tf': dt})
    #         return F(x0=p, p=u)['xf']
    #
    #     else:
    #         raise ValueError(f"Unknown integration method: {integration_method}")

    def build_integrator(method):
        def build_rk4_step(p, u, dt):
            # This is the exact RK4 formula you had before, but using symbolic variables
            k1 = dynamics(p, u)
            k2 = dynamics(p + dt / 2 * k1, u)
            k3 = dynamics(p + dt / 2 * k2, u)
            k4 = dynamics(p + dt * k3, u)
            return p + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        p_sym = ca.MX.sym('p', n_states)  # Current state
        u_sym = ca.MX.sym('u', n_controls)  # Control input
        dt_sym = ca.MX.sym('dt', 1)  # Time step
        rk4_integrator = ca.Function('rk4_integrator', [p_sym, u_sym, dt_sym], [build_rk4_step(p_sym, u_sym, dt_sym)])

        return rk4_integrator

    # p = ca.MX.sym('p', n_states)
    # u = ca.MX.sym('u', n_controls)
    # ode_rhs = dynamics(p, u)
    # integrator = ca.integrator('F', 'tools', {
    #     'x': p, 'p': u, 'ode': ode_rhs
    # }, {
    #     'tf': dt, 'scheme': 'rk4'
    # })

    # Setup optimization problem
    opti = ca.Opti()

    # Decision variables
    P = opti.variable(n_states, N + 1)
    U = opti.variable(n_controls, N)

    my_integrator = build_integrator(integration_method)

    # Dynamics constraints
    for k in range(N):
        p_next = my_integrator(P[:, k], U[:, k], dt)
        opti.subject_to(P[:, k+1] == p_next)

    # Path constraints
    opti.subject_to(ca.vec(P) >= -1e-6)  # Non-negative probabilities
    # for k in range(N + 1):
    #     opti.subject_to(ca.sum1(P[:, k]) <= 1.0 + 1e-6)  # Probability conservation

    # Control bounds
    opti.subject_to(ca.vec(U) >= control_bounds[0])
    opti.subject_to(ca.vec(U) <= control_bounds[1])

    # Initial condition
    opti.subject_to(P[:, 0] == p_initial)

    # Objective function
    objective = 0
    for k in range(N):
        objective += running_cost(P[:, k], U[:, k])
    objective += terminal_cost(P[:, N])

    opti.minimize(objective)

    # Initial guess
    if initial_guess is not None:
        opti.set_initial(P, initial_guess['P'])
        opti.set_initial(U, initial_guess['U'])
    else:
        # Default: uniform distribution
        opti.set_initial(P, np.ones((n_states, N + 1)) / n_states)
        opti.set_initial(U, np.zeros((n_controls, N)))

    # Solver configuration
    default_options = {
        'ipopt.print_level': 5 if verbose else 0,
        'print_time': 1 if verbose else 0,
        'ipopt.max_iter': 2000,
        'ipopt.tol': 1e-6,
    }
    if solver_options is not None:
        default_options.update(solver_options)

    opti.solver('ipopt', default_options)

    # Solve
    if verbose:
        print("=" * 60)
        print("SOLVING KOLMOGOROV OPTIMAL CONTROL")
        print("=" * 60)
        print(f"States: {n_states}, Controls: {n_controls}")
        print(f"Time horizon: {T}s, Steps: {N}")
        print(f"Integration: {integration_method}")
        print(f"Decision variables: {n_states * (N + 1) + n_controls * N}")
        print("=" * 60)

    try:
        sol = opti.solve()
        P_opt = sol.value(P)
        U_opt = sol.value(U)
        cost_opt = sol.value(objective)
        success = True
        if verbose:
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print(f"Optimal cost: {cost_opt:.6f}")
            print("=" * 60)
    except RuntimeError as e:
        if verbose:
            print(f"\nSolver failed: {e}")
            print("Returning debug solution...")
        P_opt = opti.debug.value(P)
        U_opt = opti.debug.value(U)
        cost_opt = opti.debug.value(objective)
        success = False

    # Return results
    t = np.linspace(0, T, N + 1)

    return {
        'P': P_opt,
        'U': U_opt,
        't': t,
        'cost': cost_opt,
        'success': success,
        'opti': opti,
        'dt': dt,
        'N': N
    }


# ============================================================================
# Visualization Helper
# ============================================================================

def plot_kolmogorov_solution(result, p_target=None, figsize=(14, 10)):
    """Plot the optimal control solution."""
    P = result['P']
    U = result['U']
    t = result['t']
    n_states, _ = P.shape
    n_controls, N = U.shape

    fig = plt.figure(figsize=figsize)

    # Plot 1: Probability evolution
    ax1 = plt.subplot(3, 1, 1)
    for i in range(n_states):
        ax1.plot(t, P[i, :], linewidth=2, label=f'State {i}')
    if p_target is not None:
        for i, val in enumerate(p_target):
            if val > 0:
                ax1.axhline(y=val, color='r', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Optimal Probability Distribution Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])

    # Plot 2: Control signals
    ax2 = plt.subplot(3, 1, 2)
    t_u = t[:-1]
    for i in range(n_controls):
        ax2.plot(t_u, U[i, :], linewidth=2, label=f'u_{i}')
    ax2.set_ylabel('Control', fontsize=12)
    ax2.set_title('Optimal Control Signals', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heatmap
    ax3 = plt.subplot(3, 1, 3)
    im = ax3.imshow(P, aspect='auto', cmap='viridis', interpolation='nearest',
                    extent=[0, t[-1], -0.5, n_states - 0.5], origin='lower')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('State', fontsize=12)
    ax3.set_title('Probability Distribution Heatmap', fontsize=12)
    ax3.set_yticks(range(n_states))
    plt.colorbar(im, ax=ax3, label='Probability')

    plt.tight_layout()
    return fig


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":

    # Problem setup
    n_states = 150
    n_controls = 2


    # Define rate function (SIR model)

    def rate(i, j, p, u, n_states):
        if j - i == 1:
            # Option 1: Proper mass action
            return (1 + i * u[0]) * p[j]
        elif j - i == -1:
            return (1 + i * u[1]) * p[j]
        else:
            return 0.0

    def running_cost(p, u):
        return - 10 * (1 - p[n_states - 1] - p[0]) / n_states + ca.sumsqr(u) / n_states


    def terminal_cost(p):
        return -100 * p[n_states - 1]


    # Initial condition
    p0 = np.ones(n_states)/n_states

    # Solve
    result = solve_kolmogorov_optimal_control(
        rate_function=rate,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        n_states=n_states,
        n_controls=n_controls,
        p_initial=p0,
        T=5.0,
        N=500,
        control_bounds=(0.0, 15.0),
        integration_method='rk4',
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial: {result['P'][:, 0]}")
    print(f"Final:   {result['P'][:, -1]}")
    print(f"Prob conservation: {np.sum(result['P'][:, -1]):.8f}")
    print("=" * 60)

    # Plot
    fig = plot_kolmogorov_solution(result)
    plt.show()
