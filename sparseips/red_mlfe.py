"""
Reduced Local-Field Equations (RLFE) solver for epidemic-type IPS.

Implements Theorem 2.12 and equations (2.10)-(2.13) from:
  Cocomello, Li, Ramanan (2025)
  "A class of interacting particle systems for which the Markov
   local-field equations are exact."

Applicable to IPS on sparse random graphs (Erdős–Rényi or configuration
model) whose state space has state 0 as an absorbing susceptible state
(no reinfection) and states 1..m as post-infection states.

---

Vertex-type extension
---------------------
When ``vertex_type_init`` is supplied, the RLFE tracks n_τ² conditional
distributions  f^{r,n}(t)  (one per root-type r × neighbour-type n pair).

Infection rates have Kronecker product structure:
    b^{r,n}_{i→j}  =  B_type[r, n]  *  b_base[i, j]

where ``type_interaction`` is the n_τ × n_τ matrix B_type (default: all-ones,
i.e. type-independent infection).  Each root type r has its own cumulative
pressure  F^r(t) = ∫ f_dot_b^r ds  and corresponding  φ^r = Φ(F^r).

ODE dimension (polynomial in n, n_k, n_τ):
    n_τ² n  +  n_τ  +  n_τ n_k n  +  n_τ m²

---

Edge-type extension
-------------------
When ``edge_type_init`` is supplied, the RLFE tracks n_e conditional
distributions  f^e(t)  (one per edge type e).

Infection rates scale with edge type:
    b^e_{i→j}  =  w_e  *  b_base[i, j]

where ``edge_weights`` is the n_e vector w.  There is a single cumulative
pressure F(t) = ∫ f_dot_b ds and a single φ = Φ(F), because all roots
experience the same mixture of edge types regardless of their own state.
The inflow for a susceptible neighbour is also type-e independent: a
neighbour's children have edge types drawn from the same distribution π,
so the rate at which a susceptible neighbour gets infected by its children
is identical for all edge types e.

ODE dimension:
    n_e n  +  1  +  n_k n  +  m²
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
        vertex_type_init: dict = None,
        type_interaction: np.ndarray = None,
        edge_type_init: dict = None,
        edge_weights: np.ndarray = None,
        solver_type: str = 'explicit',
        step_control: str = 'adaptive',
        verbose: bool = True,
        throw: bool = True,
        return_per_type: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate the Reduced Local-Field Equation (RLFE) from Theorem 2.12.

    Parameters
    ----------
    ips : ParticleSystem
        Must carry ``ips.params['b']`` (infection-rate matrix) and
        ``ips.params['rho']`` (autonomous-transition matrix).
        State index 0 must be the susceptible state.
    initial_conditions : dict[state, float]
        Marginal probability of each state at t = 0.
    max_time : float
    num_grid_points : int
    vertex_type_init : dict[type_label, float] or None
        Fraction of each vertex type, e.g. ``{'v1': 0.1, 'v2': 0.9}``.
        Requires ``ips.vertex_type_space`` and
        ``ips.params['rho_per_type']`` (shape n_τ × n × n).
        Cannot be combined with ``edge_type_init``.
    type_interaction : np.ndarray, shape (n_τ, n_τ), optional
        Type-interaction matrix B_type.  The infection rate from a
        type-n neighbour (state i) to a type-r susceptible (→ state j) is
        ``B_type[r, n] * b_base[i, j]``.
        Defaults to all-ones (type-independent infection).
    edge_type_init : dict[type_label, float] or None
        Fraction of each edge type, e.g. ``{'h': 0.2, 'l': 0.8}``.
        Requires ``ips.edge_type_space``.  Cannot be combined with
        ``vertex_type_init``.
    edge_weights : array-like, shape (n_e,), optional
        Multiplicative weight w_e for each edge type.  The infection rate
        through a type-e edge is ``w_e * b_base[i, j]``.
        Reads ``ips.params['edge_weights']`` if available, else defaults to
        all-ones (unweighted edges).
    solver_type : str
        ``'explicit'`` (Dopri5) or ``'implicit'`` (Kvaerno3).
    step_control : str
        ``'adaptive'`` (PID controller) or ``'constant'``.
    verbose : bool
    throw : bool
        If True, raise on solver failure.

    Returns
    -------
    time : np.ndarray, shape (num_grid_points,)
    sol  : np.ndarray, shape (len(state_space), num_grid_points)
        Marginal state occupation probabilities p_t(c) aggregated over types.
    index_to_state : dict[int, any]
    """
    if ips.params is None or 'b' not in ips.params or 'rho' not in ips.params:
        raise ValueError(
            "ips must have params['b'] and params['rho']. "
            "Use a GeneralizedSIR-like ParticleSystem."
        )
    if vertex_type_init is not None and edge_type_init is not None:
        raise ValueError("vertex_type_init and edge_type_init cannot both be specified.")

    has_edge_types = edge_type_init is not None
    if has_edge_types:
        return _simulate_red_mlfe_edge(
            ips=ips,
            initial_conditions=initial_conditions,
            max_time=max_time,
            num_grid_points=num_grid_points,
            edge_type_init=edge_type_init,
            edge_weights=edge_weights,
            solver_type=solver_type,
            step_control=step_control,
            verbose=verbose,
            throw=throw,
        )

    state_space = ips.state_space
    n   = len(state_space)
    m   = n - 1
    deg_dist = ips.deg_dist
    deg_supp = sorted(k for k, p in deg_dist.items() if p > 0)
    n_k = len(deg_supp)

    # ── base rate matrices ────────────────────────────────────────────────────
    b_mat_np   = np.array(ips.params['b'],   dtype=np.float32)
    rho_mat_np = np.array(ips.params['rho'], dtype=np.float32)
    b_vec_np   = b_mat_np.sum(axis=1)          # (n,) total infectivity per state

    # ── vertex-type setup ────────────────────────────────────────────────────
    # Untyped case: treat as a single "root-type 0, neighbour-type 0" pair.
    has_types = vertex_type_init is not None
    if has_types:
        if ips.vertex_type_space is None:
            raise ValueError("vertex_type_init requires ips.vertex_type_space to be set.")
        if ips.params.get('rho_per_type') is None:
            raise ValueError(
                "vertex types require ips.params['rho_per_type'] (shape n_tau x n x n). "
                "Construct the ParticleSystem with vertex_type_space set."
            )
        type_space      = ips.vertex_type_space
        n_tau           = len(type_space)
        pi_np           = np.array([vertex_type_init[t] for t in type_space], dtype=np.float32)
        rho_per_type_np = np.array(ips.params['rho_per_type'], dtype=np.float32)  # (n_tau, n, n)
        if type_interaction is not None:
            B_type_np = np.asarray(type_interaction, dtype=np.float32)
        else:
            B_type_np = np.ones((n_tau, n_tau), dtype=np.float32)
    else:
        n_tau           = 1
        pi_np           = np.array([1.0], dtype=np.float32)
        rho_per_type_np = rho_mat_np[None, :, :]    # (1, n, n)
        B_type_np       = np.ones((1, 1), dtype=np.float32)

    # Per-type sub-generator for post-infection states (n_tau, m, m).
    rho_sub_np = np.zeros((n_tau, m, m), dtype=np.float32)
    for tau in range(n_tau):
        rsub = rho_per_type_np[tau, 1:, 1:].copy()
        rsub -= np.diag(rsub.sum(axis=1))
        rho_sub_np[tau] = rsub

    # Arrays for Φ(F) = M''_θ(−F)/M'_θ(−F) − 1
    k_phi_np  = np.array([k for k in deg_supp if k > 0], dtype=np.float32)
    th_phi_np = np.array([deg_dist[k] for k in deg_supp if k > 0], dtype=np.float32)

    # ── static JAX args ───────────────────────────────────────────────────────
    static_args = {
        'pi':      jnp.array(pi_np),                      # (n_tau,)
        'b_mat':   jnp.array(b_mat_np),                   # (n, n)
        'b_vec':   jnp.array(b_vec_np),                   # (n,)
        'B_type':  jnp.array(B_type_np),                  # (n_tau, n_tau)
        'rho_sub': jnp.array(rho_sub_np),                 # (n_tau, m, m)
        'k_arr':   jnp.array(np.array(deg_supp, dtype=np.float32)),
        'k_phi':   jnp.array(k_phi_np),
        'th_phi':  jnp.array(th_phi_np),
    }

    # ── ODE state layout ──────────────────────────────────────────────────────
    #   y[0          : n_tau²n]           → f_all   (n_tau, n_tau, n)  f^{r,n}
    #   y[n_tau²n    : n_tau²n + n_tau]   → F_all   (n_tau,)           F^r
    #   y[... + n_tau: ... + n_tau·n_k·n] → P0_all  (n_tau, n_k, n)   P0^r_{k,c}
    #   y[...        : end]               → Pa_all  (n_tau, m, m)      Pa^r_{a,c}
    _f_end   = n_tau * n_tau * n
    _F_end   = _f_end + n_tau
    _P0_end  = _F_end + n_tau * n_k * n

    def _vf(t, y, args):
        pi      = args['pi']       # (n_tau,)
        b_mat   = args['b_mat']    # (n, n)
        b_vec   = args['b_vec']    # (n,)
        B_type  = args['B_type']   # (n_tau, n_tau)
        rho_sub = args['rho_sub']  # (n_tau, m, m)
        k_arr   = args['k_arr']
        k_phi   = args['k_phi']
        th_phi  = args['th_phi']

        f_all  = y[:_f_end].reshape(n_tau, n_tau, n)          # (n_tau, n_tau, n)
        F_all  = y[_f_end:_F_end]                              # (n_tau,)
        P0_all = y[_F_end:_P0_end].reshape(n_tau, n_k, n)     # (n_tau, n_k, n)
        Pa_all = y[_P0_end:].reshape(n_tau, m, m)              # (n_tau, m, m)

        # f_dot_b^r = Σ_n π_n B_type[r,n] (b_vec · f^{r,n})
        fdotb_rn = jnp.einsum('i,rni->rn', b_vec, f_all)               # (n_tau, n_tau)
        f_dot_b  = jnp.einsum('n,rn->r', pi, B_type * fdotb_rn)        # (n_tau,)

        # Φ^r(F^r) = M''_θ(−F^r)/M'_θ(−F^r) − 1  (one per root type)
        exp_neg = jnp.exp(-k_phi[None, :] * F_all[:, None])             # (n_tau, |k_phi|)
        M1  = jnp.einsum('p,rp->r', k_phi * th_phi,          exp_neg)   # (n_tau,)
        M2  = jnp.einsum('p,rp->r', k_phi * k_phi * th_phi,  exp_neg)   # (n_tau,)
        phi = jnp.where(M1 > 1e-12, M2 / M1 - 1.0, 0.0)                # (n_tau,)

        # f_b_T[r, n, a] = (b_mat.T @ f^{r,n})[a] = Σ_i b_mat[i,a] f[r,n,i]
        f_b_T = jnp.einsum('ia,rni->rna', b_mat, f_all)                 # (n_tau, n_tau, n)

        # infection_inflow[r,n,a]: rate at which type-n non-root (sus.) is
        # infected into state a by a random child (averaged over child types).
        #   = Σ_{n'} π_{n'} B_type[n,n'] f_b_T[r,n',a]
        infection_inflow = jnp.einsum('p,np,rpa->rna', pi, B_type, f_b_T)   # (n_tau, n_tau, n)

        # P0_inflow[r,a]: rate for type-r root susceptible infected into a
        # by one random neighbour (used for P0 equations).
        #   = Σ_{n'} π_{n'} B_type[r,n'] f_b_T[r,n',a]
        P0_inflow = jnp.einsum('p,rp,rpa->ra', pi, B_type, f_b_T)           # (n_tau, n)

        # ── df^{r,n} equations (2.10, generalised) ───────────────────────────
        # df^{r,n}_0 = f_dot_b^r · f^{r,n}_0 · (1 − φ^r)
        df_0 = f_dot_b[:, None] * f_all[:, :, 0] * (1.0 - phi[:, None])     # (n_tau, n_tau)

        # df^{r,n}_a = f^{r,n}_0 φ^r inflow[r,n,a]
        #            + (ρ^n)^T f^{r,n}_{1:}
        #            + f^{r,n}_{1:} (f_dot_b^r − B[r,n] b_vec[a])
        inf_term   = (f_all[:, :, 0:1] * phi[:, None, None]
                      * infection_inflow[:, :, 1:])                          # (n_tau, n_tau, m)
        rho_term   = jnp.einsum('nba,rnb->rna', rho_sub, f_all[:, :, 1:])   # (n_tau, n_tau, m)
        scale_term = (f_all[:, :, 1:]
                      * (f_dot_b[:, None, None]
                         - B_type[:, :, None] * b_vec[None, None, 1:]))      # (n_tau, n_tau, m)
        df_a  = inf_term + rho_term + scale_term                              # (n_tau, n_tau, m)
        df_all_new = jnp.concatenate([df_0[:, :, None], df_a], axis=2)       # (n_tau, n_tau, n)

        # ── P0^r equations (2.13) ─────────────────────────────────────────────
        dP0_0 = -f_dot_b[:, None] * k_arr[None, :] * P0_all[:, :, 0]        # (n_tau, n_k)
        dP0_a = (P0_all[:, :, 0:1] * k_arr[None, :, None]
                 * P0_inflow[:, None, 1:]
                 + jnp.einsum('rki,rij->rkj', P0_all[:, :, 1:], rho_sub))   # (n_tau, n_k, m)
        dP0 = jnp.concatenate([dP0_0[:, :, None], dP0_a], axis=2)           # (n_tau, n_k, n)

        # ── Pa^r equations (2.13) ─────────────────────────────────────────────
        dPa = jnp.einsum('rai,rij->raj', Pa_all, rho_sub)                    # (n_tau, m, m)

        return jnp.concatenate([
            df_all_new.ravel(),
            f_dot_b,                # dF^r/dt = f_dot_b^r
            dP0.ravel(),
            dPa.ravel(),
        ])

    # ── initial conditions ────────────────────────────────────────────────────
    p0_np = np.array([initial_conditions[s] for s in state_space], dtype=np.float32)

    # f^{r,n}(0) = p0 for all (r,n): types and initial states are independent
    f_all_init  = np.tile(p0_np, (n_tau, n_tau, 1)).astype(np.float32)   # (n_tau, n_tau, n)
    F_all_init  = np.zeros(n_tau, dtype=np.float32)
    P0_all_init = np.zeros((n_tau, n_k, n), dtype=np.float32)
    P0_all_init[:, :, 0] = 1.0                                            # P0^r_{k,0}(0) = 1
    Pa_all_init = np.stack([np.eye(m, dtype=np.float32)] * n_tau)         # (n_tau, m, m)

    y0 = jnp.array(np.concatenate([
        f_all_init.ravel(),
        F_all_init,
        P0_all_init.ravel(),
        Pa_all_init.ravel(),
    ]))

    # ── diffrax solver setup ──────────────────────────────────────────────────
    term = diffrax.ODETerm(_vf)

    if solver_type == 'explicit':
        solver = diffrax.Dopri5()
    elif solver_type == 'implicit':
        linear_solver = lx.GMRES(rtol=1e-2, atol=1e-2, restart=20)
        root_finder   = optx.Newton(rtol=1e-3, atol=1e-3, linear_solver=linear_solver)
        solver        = diffrax.Kvaerno3(root_finder=root_finder)
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
        last_t   = float(valid_ts.max()) if len(valid_ts) > 0 else 0.0
        print(f'  Warning: RLFE solver did not converge; '
              f'partial solution up to t={last_t:.3g} / {max_time}')

    # ── reconstruct marginals ─────────────────────────────────────────────────
    # p_t(c) = Σ_r π_r [p₀(0) Σ_k θ(k) P0^r_{k,c}(t)  +  Σ_{a≥1} p₀(a) Pa^r_{a,c}(t)]
    ys   = np.array(sol.ys)
    nt   = ys.shape[0]
    P0_t = ys[:, _F_end:_P0_end].reshape(nt, n_tau, n_k, n)   # (nt, n_tau, n_k, n)
    Pa_t = ys[:, _P0_end:].reshape(nt, n_tau, m, m)            # (nt, n_tau, m, m)

    theta_k = np.array([deg_dist[k] for k in deg_supp], dtype=np.float64)
    p0      = p0_np.astype(np.float64)
    pi      = pi_np.astype(np.float64)

    prob = p0[0] * np.einsum('r,k,trkc->ct', pi, theta_k, P0_t)   # (n, nt)
    if m > 0:
        prob[1:, :] += np.einsum('r,a,trac->ct', pi, p0[1:], Pa_t)

    index_to_state = {i: s for i, s in enumerate(state_space)}
    if return_per_type:
        # prob_per_type[r, c, t] = P(state = c | type = r) at each time step
        prob_per_type = p0[0] * np.einsum('k,trkc->rct', theta_k, P0_t)  # (n_tau, n, nt)
        if m > 0:
            prob_per_type[:, 1:, :] += np.einsum('a,trac->rct', p0[1:], Pa_t)
        return np.array(sol.ts), prob, prob_per_type, index_to_state
    return np.array(sol.ts), prob, index_to_state


# ─────────────────────────────────────────────────────────────────────────────
# Edge-typed RLFE
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_red_mlfe_edge(
        ips: ParticleSystem,
        initial_conditions: dict,
        max_time: float,
        num_grid_points: int,
        edge_type_init: dict,
        edge_weights,
        solver_type: str,
        step_control: str,
        verbose: bool,
        throw: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Edge-typed RLFE.  Called internally by simulate_red_mlfe when
    edge_type_init is supplied.

    ODE state layout:
        y[0        : n_e*n]           f_all  (n_e, n)   f^e
        y[n_e*n]                      F      scalar      cumulative pressure
        y[n_e*n+1  : n_e*n+1+n_k*n]  P0     (n_k, n)   P0_{k,c}
        y[...      : end]             Pa     (m, m)      Pa_{a,c}
    """
    state_space = ips.state_space
    n   = len(state_space)
    m   = n - 1
    deg_dist = ips.deg_dist
    deg_supp = sorted(k for k, p in deg_dist.items() if p > 0)
    n_k = len(deg_supp)

    if ips.edge_type_space is None:
        raise ValueError("edge_type_init requires ips.edge_type_space to be set.")
    edge_type_space = ips.edge_type_space
    n_e = len(edge_type_space)

    pi_e_np = np.array([edge_type_init[e] for e in edge_type_space], dtype=np.float32)

    if edge_weights is not None:
        w_np = np.asarray(edge_weights, dtype=np.float32)
        if w_np.shape != (n_e,):
            raise ValueError(f"edge_weights must have shape ({n_e},), got {w_np.shape}")
    elif ips.params.get('edge_weights') is not None:
        w_np = np.array(ips.params['edge_weights'], dtype=np.float32)
    else:
        w_np = np.ones(n_e, dtype=np.float32)

    b_mat_np   = np.array(ips.params['b'],   dtype=np.float32)
    rho_mat_np = np.array(ips.params['rho'], dtype=np.float32)
    b_vec_np   = b_mat_np.sum(axis=1)    # (n,) total infectivity per state

    # Sub-generator for post-infection states (m × m)
    rho_sub_np = rho_mat_np[1:, 1:].copy()
    rho_sub_np -= np.diag(rho_sub_np.sum(axis=1))

    k_phi_np  = np.array([k for k in deg_supp if k > 0], dtype=np.float32)
    th_phi_np = np.array([deg_dist[k] for k in deg_supp if k > 0], dtype=np.float32)

    # ODE index boundaries
    _f_end  = n_e * n
    _F_end  = _f_end + 1
    _P0_end = _F_end + n_k * n

    static_args = {
        'pi_e':    jnp.array(pi_e_np),                              # (n_e,)
        'w':       jnp.array(w_np),                                 # (n_e,)
        'b_mat':   jnp.array(b_mat_np),                             # (n, n)
        'b_vec':   jnp.array(b_vec_np),                             # (n,)
        'rho_sub': jnp.array(rho_sub_np),                           # (m, m)
        'k_arr':   jnp.array(np.array(deg_supp, dtype=np.float32)), # (n_k,)
        'k_phi':   jnp.array(k_phi_np),                             # (|k_phi|,)
        'th_phi':  jnp.array(th_phi_np),                            # (|k_phi|,)
    }

    def _vf(t, y, args):
        pi_e    = args['pi_e']    # (n_e,)
        w       = args['w']       # (n_e,)
        b_mat   = args['b_mat']   # (n, n)
        b_vec   = args['b_vec']   # (n,)
        rho_sub = args['rho_sub'] # (m, m)
        k_arr   = args['k_arr']   # (n_k,)
        k_phi   = args['k_phi']
        th_phi  = args['th_phi']

        f_all = y[:_f_end].reshape(n_e, n)            # (n_e, n)
        F     = y[_f_end]                              # scalar
        P0    = y[_F_end:_P0_end].reshape(n_k, n)     # (n_k, n)
        Pa    = y[_P0_end:].reshape(m, m)              # (m, m)

        # Total infection pressure (one per edge, mixed over edge types)
        # f_dot_b = Σ_e π_e w_e (b_vec · f^e)
        f_dot_b = jnp.dot(pi_e * w, f_all @ b_vec)    # scalar

        # φ(F) = M''_θ(−F)/M'_θ(−F) − 1
        exp_neg = jnp.exp(-k_phi * F)
        M1  = jnp.dot(k_phi * th_phi,         exp_neg)
        M2  = jnp.dot(k_phi * k_phi * th_phi, exp_neg)
        phi = jnp.where(M1 > 1e-12, M2 / M1 - 1.0, 0.0)  # scalar

        # f_b_T[e, a] = Σ_i b_mat[i,a] f^e_i
        f_b_T = jnp.einsum('ia,ei->ea', b_mat, f_all)  # (n_e, n)

        # infection_inflow[a] = Σ_e π_e w_e f_b_T[e, a]
        # This is type-e independent: a neighbour's children share the
        # same edge-type distribution π regardless of which edge connects
        # the neighbour to the root.
        infection_inflow = jnp.einsum('e,ea->a', pi_e * w, f_b_T)  # (n,)

        # ── df^e equations ───────────────────────────────────────────────────
        # df^e_0 = f_dot_b · f^e_0 · (1 − φ)
        df_0 = f_dot_b * f_all[:, 0] * (1.0 - phi)            # (n_e,)

        # df^e_a (a ≥ 1):
        #   = f^e_0 φ infection_inflow[a]
        #   + Σ_b rho_sub[b,a] f^e_b
        #   + f^e_a (f_dot_b − w_e b_vec[a])
        inf_term   = f_all[:, 0:1] * phi * infection_inflow[None, 1:]          # (n_e, m)
        rho_term   = jnp.einsum('ba,eb->ea', rho_sub, f_all[:, 1:])            # (n_e, m)
        scale_term = f_all[:, 1:] * (f_dot_b - w[:, None] * b_vec[None, 1:])  # (n_e, m)
        df_a = inf_term + rho_term + scale_term                                 # (n_e, m)

        df_all = jnp.concatenate([df_0[:, None], df_a], axis=1)  # (n_e, n)

        # ── P0 equations ─────────────────────────────────────────────────────
        # dP0_{k,0} = −f_dot_b · k · P0_{k,0}
        # dP0_{k,a} = P0_{k,0} · k · infection_inflow[a]  +  rho_sub.T @ P0_{k,1:}
        dP0_0 = -f_dot_b * k_arr * P0[:, 0]                           # (n_k,)
        dP0_a = (P0[:, 0:1] * k_arr[:, None] * infection_inflow[None, 1:]
                 + jnp.einsum('ki,ij->kj', P0[:, 1:], rho_sub))       # (n_k, m)
        dP0   = jnp.concatenate([dP0_0[:, None], dP0_a], axis=1)      # (n_k, n)

        # ── Pa equation ───────────────────────────────────────────────────────
        dPa = jnp.einsum('ai,ij->aj', Pa, rho_sub)                     # (m, m)

        return jnp.concatenate([
            df_all.ravel(),
            jnp.array([f_dot_b]),   # dF/dt = f_dot_b
            dP0.ravel(),
            dPa.ravel(),
        ])

    # ── initial conditions ────────────────────────────────────────────────────
    p0_np = np.array([initial_conditions[s] for s in state_space], dtype=np.float32)

    f_all_init = np.tile(p0_np, (n_e, 1)).astype(np.float32)  # (n_e, n)
    F_init     = np.zeros(1, dtype=np.float32)
    P0_init    = np.zeros((n_k, n), dtype=np.float32)
    P0_init[:, 0] = 1.0
    Pa_init    = np.eye(m, dtype=np.float32)

    y0 = jnp.array(np.concatenate([
        f_all_init.ravel(),
        F_init,
        P0_init.ravel(),
        Pa_init.ravel(),
    ]))

    # ── solver ────────────────────────────────────────────────────────────────
    term = diffrax.ODETerm(_vf)

    if solver_type == 'explicit':
        solver = diffrax.Dopri5()
    elif solver_type == 'implicit':
        linear_solver = lx.GMRES(rtol=1e-2, atol=1e-2, restart=20)
        root_finder   = optx.Newton(rtol=1e-3, atol=1e-3, linear_solver=linear_solver)
        solver        = diffrax.Kvaerno3(root_finder=root_finder)
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
        print('**** Running edge-typed RLFE simulation ****')

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
        last_t   = float(valid_ts.max()) if len(valid_ts) > 0 else 0.0
        print(f'  Warning: edge-typed RLFE solver did not converge; '
              f'partial solution up to t={last_t:.3g} / {max_time}')

    # ── reconstruct marginals ─────────────────────────────────────────────────
    # p_t(c) = p₀(0) Σ_k θ_k P0_{k,c}(t)  +  Σ_{a≥1} p₀(a) Pa_{a,c}(t)
    ys   = np.array(sol.ys)
    nt   = ys.shape[0]
    P0_t = ys[:, _F_end:_P0_end].reshape(nt, n_k, n)  # (nt, n_k, n)
    Pa_t = ys[:, _P0_end:].reshape(nt, m, m)           # (nt, m, m)

    theta_k = np.array([deg_dist[k] for k in deg_supp], dtype=np.float64)
    p0      = p0_np.astype(np.float64)

    prob = p0[0] * np.einsum('k,tkc->ct', theta_k, P0_t)   # (n, nt)
    if m > 0:
        prob[1:, :] += np.einsum('a,tac->ct', p0[1:], Pa_t)

    index_to_state = {i: s for i, s in enumerate(state_space)}
    return np.array(sol.ts), prob, index_to_state
