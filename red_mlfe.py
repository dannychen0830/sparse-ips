"""
Reduced Local-Field Equations (RLFE) solver for epidemic-type IPS.

Implements Theorem 2.12 and equations (2.10)-(2.13) from:
  Cocomello, Li, Ramanan (2025)
  "A class of interacting particle systems for which the Markov
   local-field equations are exact."

Applicable to IPS on sparse random graphs (Erdős–Rényi or configuration
model) whose state space has state 0 as an absorbing susceptible state
(no reinfection) and states 1..m as post-infection states.

The RLFE system dimension grows as  (m+1) + 1 + (m+1)|supp(θ)| + m²
rather than exponentially in degree, making it far cheaper than the MLFE.
"""

import numpy as np
import jax.numpy as jnp
import diffrax
import lineax as lx
import optimistix as optx

from sparseips.ips_class import ParticleSystem


def simulate_red_mlfe(
        ips: ParticleSystem,
        initial_conditions: dict,
        max_time: float,
        num_grid_points: int = 100,
        solver_type: str = 'explicit',
        step_control: str = 'adaptive',
        verbose: bool = True,
        throw: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate the Reduced Local-Field Equation (RLFE) from Theorem 2.12.

    Parameters
    ----------
    ips : ParticleSystem
        Must carry ips.params['b'] (infection-rate matrix) and
        ips.params['rho'] (autonomous-transition matrix) in the format
        used by GeneralizedSIR.  State index 0 must be susceptible.
    initial_conditions : dict[state, float]
        Marginal probability of each state at t = 0.
    max_time : float
    num_grid_points : int
    solver_type : str
        'explicit' (Dopri5) or 'implicit' (Kvaerno3).
    step_control : str
        'adaptive' (PID controller) or 'constant'.
    verbose : bool
    throw : bool
        If True, raise on solver failure; if False, return partial solution.

    Returns
    -------
    time : np.ndarray, shape (num_grid_points,)
    sol  : np.ndarray, shape (len(state_space), num_grid_points)
        Marginal state occupation probabilities p_t(c) for each state c.
    index_to_state : dict[int, any]
        Maps integer index → state label (same convention as simulate_markov_lfe).
    """
    if ips.params is None or 'b' not in ips.params or 'rho' not in ips.params:
        raise ValueError(
            "ips must have params['b'] and params['rho']. "
            "Use a GeneralizedSIR-like ParticleSystem."
        )

    state_space = ips.state_space
    n   = len(state_space)     # total states  (0..m)
    m   = n - 1                # post-infection states
    deg_dist = ips.deg_dist
    deg_supp = sorted(k for k, p in deg_dist.items() if p > 0)
    n_k = len(deg_supp)

    # ── rate matrices (float32 to match rest of codebase) ────────────────────
    b_mat_np   = np.array(ips.params['b'],   dtype=np.float32)
    rho_mat_np = np.array(ips.params['rho'], dtype=np.float32)

    # b_vec[i] = Σ_j b_mat[i,j] : total infectivity of state i (0 for susceptible)
    b_vec_np = b_mat_np.sum(axis=1)

    # Full generator for the post-infection sub-space (diagonal = −row sum)
    rho_sub_np = rho_mat_np[1:, 1:].copy()
    rho_sub_np -= np.diag(rho_sub_np.sum(axis=1))

    # Arrays for Φ(F) = M''_θ(−F)/M'_θ(−F) − 1 : only nonzero k contribute
    k_phi_np  = np.array([k for k in deg_supp if k > 0], dtype=np.float32)
    th_phi_np = np.array([deg_dist[k] for k in deg_supp if k > 0], dtype=np.float32)

    # ── build static args (all JAX arrays) ───────────────────────────────────
    static_args = {
        'b_mat':   jnp.array(b_mat_np),
        'b_vec':   jnp.array(b_vec_np),
        'rho_sub': jnp.array(rho_sub_np),
        'k_arr':   jnp.array(np.array(deg_supp, dtype=np.float32)),
        'k_phi':   jnp.array(k_phi_np),
        'th_phi':  jnp.array(th_phi_np),
    }

    # ── vector field (closure captures static integer dims) ──────────────────
    # n, m, n_k are Python ints captured at definition time.  Inside a
    # JAX-traced function, compile-time-constant integers can be used freely
    # for slicing and reshaping without tracing overhead.

    def _vf(t, y, args):
        """
        ODE state-vector layout (total size = n + 1 + n*n_k + m*m):

            y[0:n]                  → f[0..m]       (conditional state probs)
            y[n]                    → F             (cumulative f·b integral)
            y[n+1 : n+1+n*n_k]     → P0 (n_k × n)  P_{0,c;k}
            y[n+1+n*n_k : ...]      → Pa (m  × m)   P_{a,c}, a,c ∈ {1..m}
        """
        f  = y[:n]
        F  = y[n]
        P0 = y[n + 1 : n + 1 + n * n_k].reshape(n_k, n)
        Pa = y[n + 1 + n * n_k :].reshape(m, m)

        b_mat   = args['b_mat']
        b_vec   = args['b_vec']
        rho_sub = args['rho_sub']
        k_arr   = args['k_arr']
        k_phi   = args['k_phi']
        th_phi  = args['th_phi']

        # f · b = Σ_ℓ f_ℓ b_ℓ
        f_dot_b = jnp.dot(f, b_vec)

        # Φ(F) = M''_θ(−F) / M'_θ(−F) − 1  (eq. 2.7)
        exp_neg = jnp.exp(-k_phi * F)
        M1 = jnp.dot(k_phi * th_phi, exp_neg)
        M2 = jnp.dot(k_phi * k_phi * th_phi, exp_neg)
        phi = jnp.where(M1 > 1e-12, M2 / M1 - 1.0, 0.0)

        # Σ_ℓ ω_{ℓ,a} f_ℓ for each target state a
        infection_inflow = b_mat.T @ f   # (n,)

        # --- f equations (2.10) ---
        df_0 = f_dot_b * f[0] * (1.0 - phi)
        df_a = (f[0] * phi * infection_inflow[1:]
                + rho_sub.T @ f[1:]
                + f[1:] * (f_dot_b - b_vec[1:]))
        df   = jnp.concatenate([df_0.reshape(1), df_a])

        # --- P₀ equations (2.13) ---
        dP0_0 = -f_dot_b * k_arr * P0[:, 0]                               # (n_k,)
        dP0_a = (P0[:, 0:1] * k_arr[:, None] * infection_inflow[None, 1:]
                 + P0[:, 1:] @ rho_sub)                                    # (n_k, m)
        dP0   = jnp.concatenate([dP0_0[:, None], dP0_a], axis=1)          # (n_k, n)

        # --- Pₐ equations (2.13) ---
        dPa = Pa @ rho_sub   # (m, m)

        return jnp.concatenate([df, f_dot_b.reshape(1), dP0.ravel(), dPa.ravel()])

    # ── initial conditions ────────────────────────────────────────────────────
    p0_np = np.array([initial_conditions[s] for s in state_space], dtype=np.float32)

    P0_init = np.zeros((n_k, n), dtype=np.float32)
    P0_init[:, 0] = 1.0          # P_{0,c;k}(0) = 1_{c=0}  for all k

    Pa_init = np.eye(m, dtype=np.float32)  # P_{a,c}(0) = δ_{a,c}

    y0 = jnp.array(np.concatenate([
        p0_np,
        [0.0],            # F(0) = 0
        P0_init.ravel(),
        Pa_init.ravel(),
    ]))

    # ── diffrax solver setup ──────────────────────────────────────────────────
    term = diffrax.ODETerm(_vf)

    if solver_type == 'explicit':
        solver = diffrax.Dopri5()
    elif solver_type == 'implicit':
        linear_solver = lx.GMRES(rtol=1e-2, atol=1e-2, restart=20)
        root_finder = optx.Newton(rtol=1e-3, atol=1e-3, linear_solver=linear_solver)
        solver = diffrax.Kvaerno3(root_finder=root_finder)
    else:
        raise ValueError(f'Unknown solver_type: {solver_type!r}')

    if step_control == 'adaptive':
        stepsize_controller = diffrax.PIDController(rtol=1e-9, atol=1e-12)
    elif step_control == 'constant':
        stepsize_controller = diffrax.ConstantStepSize()
    else:
        raise ValueError(f'Unknown step_control: {step_control!r}')

    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, max_time, num_grid_points))

    if verbose:
        print('**** Running RLFE simulation ****')

    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=max_time, dt0=0.01,
        y0=y0, args=static_args,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=100000,
        throw=throw,
        progress_meter=diffrax.TqdmProgressMeter() if verbose else diffrax.NoProgressMeter(),
    )

    if not throw and sol.result != diffrax.RESULTS.successful:
        valid_ts = sol.ts[~jnp.isinf(sol.ts)]
        last_t = float(valid_ts.max()) if len(valid_ts) > 0 else 0.0
        print(f'  Warning: RLFE solver did not converge; '
              f'partial solution up to t={last_t:.3g} / {max_time}')

    # ── reconstruct marginals from equation (2.12) ───────────────────────────
    # p_t(c) = p₀(0) Σ_k θ(k) P_{0,c;k}(t)  +  Σ_{a=1}^m p₀(a) P_{a,c}(t)
    ys  = np.array(sol.ys)                               # (nt, total_size)
    nt  = ys.shape[0]
    P0_t = ys[:, n + 1 : n + 1 + n * n_k].reshape(nt, n_k, n)
    Pa_t = ys[:, n + 1 + n * n_k :].reshape(nt, m, m)

    theta_k = np.array([deg_dist[k] for k in deg_supp], dtype=np.float64)
    p0      = p0_np.astype(np.float64)

    # p₀(0) Σ_k θ(k) P_{0,c;k}  →  shape (n, nt)
    prob = p0[0] * np.einsum('k,tkc->ct', theta_k, P0_t)

    # Σ_{a≥1} p₀(a) P_{a,c}  →  shape (m, nt), added to states 1..m
    if m > 0:
        prob[1:, :] += np.einsum('a,tac->ct', p0[1:], Pa_t)

    index_to_state = {i: s for i, s in enumerate(state_space)}
    return np.array(sol.ts), prob, index_to_state
