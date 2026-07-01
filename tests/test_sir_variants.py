"""
Unit tests derived from demo/sir-and-variants/demo_sir_and_variants.py.

Each test exercises one configuration of GeneralizedSIR:
  - basic (no marks)
  - edge-marked
  - vertex-marked
  - global interaction

Tests cover:
  - Particle system construction
  - Stochastic simulation (simulate_jump_process)
  - ODE / LFE simulation (simulate_markov_lfe)
  - Basic invariants (valid states, monotone time, probability conservation)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import jax.numpy as jnp
from itertools import product

from sparseips import (
    ParticleSystem,
    simulate_jump_process,
    simulate_markov_lfe,
    get_particle_states_at_times,
)
from sparseips.util import pad_indices

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

STATE_SPACE = ['0', '1a', '1b', '2']   # S, E, I, R
LABELS = ['S', 'E', 'I', 'R']

# infection rate: exposed neighbours drive S→E
B_FULL = {('1a', '1a'): 10.0}
# autonomous transitions: E→I, I→R
RHO_FULL = {('1a', '1b'): 1.0, ('1b', '2'): 5.0}

# fill missing pairs with 0 so no KeyError inside rate()
def _fill_rates(b, rho, state_space=STATE_SPACE):
    b = dict(b)
    rho = dict(rho)
    for i in state_space:
        for j in state_space:
            b.setdefault((i, j), 0.0)
            rho.setdefault((i, j), 0.0)
    return b, rho


B, RHO = _fill_rates(B_FULL, RHO_FULL)

NUM_PARTICLES = 50          # small for fast tests
MAX_TIME = 2.0
SEED = 42
DEG_DIST = {3: 1.0}        # 3-regular graph

INITIAL_CONDITIONS_FRAC = {'0': 0.95, '1a': 0.03, '1b': 0.02, '2': 0.0}

VERTEX_TYPE_SPACE = ['v1', 'v2']
EDGE_TYPE_SPACE = ['h', 'l']
VERTEX_TYPE_INIT = {'v1': 0.1, 'v2': 0.9}
EDGE_TYPE_INIT = {'h': 0.1, 'l': 0.9}


def _make_graph(seed=SEED):
    return ParticleSystem.sample_graph_from_deg_dist(DEG_DIST, NUM_PARTICLES, seed=seed)


def _make_sim_init(graph, seed=SEED):
    """Return a per-node initial state dict sampled from fractional initial conditions."""
    np.random.seed(seed)
    probs = list(INITIAL_CONDITIONS_FRAC.values())
    return {
        i: STATE_SPACE[np.random.choice(len(STATE_SPACE), p=probs)]
        for i in graph.nodes()
    }


def _vertex_type_func():
    return 'v1' if np.random.rand() < VERTEX_TYPE_INIT['v1'] else 'v2'


def _edge_type_func():
    return 'h' if np.random.rand() < EDGE_TYPE_INIT['h'] else 'l'


# Import GeneralizedSIR from the demo (it lives outside the package)
_demo_path = os.path.join(
    os.path.dirname(__file__), '..', 'demo', 'sir-and-variants'
)
sys.path.insert(0, _demo_path)
from demo_sir_and_variants import GeneralizedSIR, cut_off  # noqa: E402


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

def assert_jumps_valid(jumps, state_space, graph):
    """Check structural invariants on a jump list."""
    valid_nodes = set(graph.nodes())
    for entry in jumps:
        node, t, (src, tgt) = entry
        assert node in valid_nodes, f"Jump from unknown node {node}"
        assert src in state_space, f"Unknown source state {src}"
        assert tgt in state_space, f"Unknown target state {tgt}"
        assert src != tgt, "Jump must change state"
        assert t >= 0, "Jump time must be non-negative"
    # jump times must be non-decreasing
    times = [t for _, t, _ in jumps]
    assert times == sorted(times), "Jump times must be non-decreasing"


def assert_lfe_valid(time, sol, idx_to_state, state_space):
    """Check ODE solution shape and probability conservation."""
    assert time.shape[0] == sol.shape[1], "time/sol column mismatch"
    state_to_index = {s: i for i, s in enumerate(state_space)}

    # marginal probability over root state must sum to ~1 at every time point
    prob = np.zeros((len(state_space), time.size))
    for idx, ode_state in idx_to_state.items():
        # ode_state is either a tuple (root, *neighbors) or
        # ((root, *neighbors), type_tuple) depending on mode
        root = ode_state[0] if isinstance(ode_state[0], str) else ode_state[0][0]
        prob[state_to_index[root], :] += sol[idx, :]

    total = prob.sum(axis=0)
    np.testing.assert_allclose(
        total, np.ones_like(total), atol=1e-3,
        err_msg="LFE marginals do not sum to 1"
    )
    # all values must be in [0, 1]
    assert np.all(sol >= -1e-6), "LFE solution has negative probabilities"


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestGeneralizedSIRConstruction:

    def test_basic_construction(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        assert ips.num_particles == NUM_PARTICLES
        assert ips.state_space == STATE_SPACE
        assert ips.deg_dist == DEG_DIST
        assert ips.global_interaction is False
        assert ips.vertex_type is None
        assert ips.edge_type is None

    def test_vertex_marked_construction(self):
        graph = _make_graph()
        np.random.seed(SEED)
        vtype = {i: _vertex_type_func() for i in graph.nodes()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            vertex_type_space=VERTEX_TYPE_SPACE,
            vertex_type=vtype,
        )
        assert ips.vertex_type_space == VERTEX_TYPE_SPACE
        assert len(ips.vertex_type) == NUM_PARTICLES

    def test_edge_marked_construction(self):
        graph = _make_graph()
        np.random.seed(SEED)
        etype = {tuple(sorted((u, v))): _edge_type_func() for u, v in graph.edges()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            edge_type_space=EDGE_TYPE_SPACE,
            edge_type=etype,
        )
        assert ips.edge_type_space == EDGE_TYPE_SPACE
        assert len(ips.edge_type) == graph.number_of_edges()

    def test_global_interaction_construction(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            global_interaction=True,
        )
        assert ips.global_interaction is True

    def test_params_are_jax_arrays(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        import jax.numpy as jnp
        assert isinstance(ips.params['b'], jnp.ndarray)
        assert isinstance(ips.params['rho'], jnp.ndarray)
        assert ips.params['b'].shape == (len(STATE_SPACE), len(STATE_SPACE))

    def test_b_matrix_values(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        si = ips.state_to_index
        # ('1a', '1a') → b[1, 1] == 10.0
        assert float(ips.params['b'][si['1a'], si['1a']]) == pytest.approx(10.0)
        # all other entries should be 0
        assert float(ips.params['b'][si['0'], si['1a']]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rate function tests
# ---------------------------------------------------------------------------

class TestRateFunction:
    """
    Tests for GeneralizedSIR.rate() using the Phase-1 unified integer-index signature:
        rate(src_int, tgt_int, neighbors_int_array, params, *, ...)
    """

    def _basic_ips(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        return ips, ips.state_to_index

    def _neighbors(self, ips, *state_strings):
        si = ips.state_to_index
        return np.array([si[s] for s in state_strings], dtype=np.int32)

    def test_susceptible_zero_rate_with_no_infected(self):
        ips, si = self._basic_ips()
        # S node with all S neighbours: no infected → rate S→E must be 0
        rate = ips.rate(si['0'], si['1a'], self._neighbors(ips, '0', '0', '0'), ips.params)
        assert float(rate) == pytest.approx(0.0)

    def test_susceptible_nonzero_rate_with_infected(self):
        ips, si = self._basic_ips()
        # S node with one E ('1a') neighbour: should have positive rate S→E
        rate = ips.rate(si['0'], si['1a'], self._neighbors(ips, '1a', '0', '0'), ips.params)
        assert float(rate) > 0.0

    def test_recovery_rate_independent_of_neighbors(self):
        ips, si = self._basic_ips()
        # Rate for E→I should not depend on neighbour states
        r1 = ips.rate(si['1a'], si['1b'], self._neighbors(ips, '0', '0', '0'), ips.params)
        r2 = ips.rate(si['1a'], si['1b'], self._neighbors(ips, '1b', '1b', '1b'), ips.params)
        assert float(r1) == pytest.approx(float(r2))

    def test_zero_rate_for_impossible_transition(self):
        ips, si = self._basic_ips()
        # R→E is not in rho, so rate should be 0
        rate = ips.rate(si['2'], si['1a'], self._neighbors(ips, '0', '0', '0'), ips.params)
        assert float(rate) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Stochastic simulation tests
# ---------------------------------------------------------------------------

class TestSimulateJumpProcess:

    def _run(self, ips, graph, seed=SEED):
        sim_init = _make_sim_init(graph, seed=seed)
        jumps = simulate_jump_process(
            ips=ips,
            initial_conditions=sim_init,
            max_time=MAX_TIME,
            seed=seed,
            verbose=False,
        )
        return jumps, sim_init

    def test_basic_sir_produces_jumps(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        jumps, _ = self._run(ips, graph)
        assert len(jumps) > 0, "Expected at least one jump in basic SIR"

    def test_basic_sir_jumps_are_valid(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        jumps, _ = self._run(ips, graph)
        assert_jumps_valid(jumps, STATE_SPACE, graph)

    def test_jumps_within_max_time(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        jumps, _ = self._run(ips, graph)
        for _, t, _ in jumps:
            assert t <= MAX_TIME, f"Jump at t={t} exceeds max_time={MAX_TIME}"

    def test_reproducibility_with_same_seed(self):
        graph = _make_graph(seed=SEED)
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        sim_init = _make_sim_init(graph, seed=SEED)
        j1 = simulate_jump_process(ips=ips, initial_conditions=sim_init, max_time=MAX_TIME, seed=SEED)
        j2 = simulate_jump_process(ips=ips, initial_conditions=sim_init, max_time=MAX_TIME, seed=SEED)
        assert len(j1) == len(j2)
        for (n1, t1, tr1), (n2, t2, tr2) in zip(j1, j2):
            assert n1 == n2 and t1 == t2 and tr1 == tr2

    def test_vertex_marked_sir_produces_valid_jumps(self):
        graph = _make_graph()
        np.random.seed(SEED)
        vtype = {i: _vertex_type_func() for i in graph.nodes()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            vertex_type_space=VERTEX_TYPE_SPACE,
            vertex_type=vtype,
        )
        jumps, _ = self._run(ips, graph)
        assert_jumps_valid(jumps, STATE_SPACE, graph)

    def test_edge_marked_sir_produces_valid_jumps(self):
        graph = _make_graph()
        np.random.seed(SEED)
        etype = {tuple(sorted((u, v))): _edge_type_func() for u, v in graph.edges()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            edge_type_space=EDGE_TYPE_SPACE,
            edge_type=etype,
        )
        jumps, _ = self._run(ips, graph)
        assert_jumps_valid(jumps, STATE_SPACE, graph)

    def test_global_interaction_sir_produces_valid_jumps(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            global_interaction=True,
        )
        jumps, _ = self._run(ips, graph)
        assert_jumps_valid(jumps, STATE_SPACE, graph)

    def test_get_particle_states_at_times(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        sim_init = _make_sim_init(graph, seed=SEED)
        jumps = simulate_jump_process(ips=ips, initial_conditions=sim_init, max_time=MAX_TIME, seed=SEED)

        timestamps = [0.0, MAX_TIME / 2, MAX_TIME]
        states_at_times = get_particle_states_at_times(jumps, sim_init, timestamps)

        assert len(states_at_times) == len(timestamps)
        for state_dict in states_at_times:
            assert set(state_dict.keys()) == set(graph.nodes())
            for state in state_dict.values():
                assert state in STATE_SPACE


# ---------------------------------------------------------------------------
# LFE (ODE) simulation tests
# ---------------------------------------------------------------------------

class TestSimulateMarkovLFE:

    def _make_basic_ips(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        return ips

    def test_basic_lfe_runs(self):
        ips = self._make_basic_ips()
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            solver_type='explicit',
            step_control='adaptive',
        )
        assert time.size == 20
        assert sol.shape[1] == 20
        assert len(idx_to_state) == sol.shape[0]

    def test_basic_lfe_probability_conservation(self):
        ips = self._make_basic_ips()
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            solver_type='explicit',
            step_control='adaptive',
        )
        assert_lfe_valid(time, sol, idx_to_state, STATE_SPACE)

    def test_basic_lfe_initial_conditions(self):
        ips = self._make_basic_ips()
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            solver_type='explicit',
            step_control='adaptive',
        )
        state_to_index = {s: i for i, s in enumerate(STATE_SPACE)}
        prob0 = np.zeros(len(STATE_SPACE))
        for idx, ode_state in idx_to_state.items():
            root = ode_state[0] if isinstance(ode_state[0], str) else ode_state[0][0]
            prob0[state_to_index[root]] += sol[idx, 0]

        # At t=0, marginals should match INITIAL_CONDITIONS_FRAC
        for state, frac in INITIAL_CONDITIONS_FRAC.items():
            np.testing.assert_allclose(
                prob0[state_to_index[state]], frac, atol=1e-3,
                err_msg=f"Initial marginal for state '{state}' is off"
            )

    def test_vertex_marked_lfe_runs(self):
        graph = _make_graph()
        np.random.seed(SEED)
        vtype = {i: _vertex_type_func() for i in graph.nodes()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            vertex_type_space=VERTEX_TYPE_SPACE,
            vertex_type=vtype,
        )
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            vertex_type_init=VERTEX_TYPE_INIT,
            solver_type='explicit',
            step_control='adaptive',
        )
        assert sol.shape[1] == 20
        assert_lfe_valid(time, sol, idx_to_state, STATE_SPACE)

    def test_edge_marked_lfe_runs(self):
        graph = _make_graph()
        np.random.seed(SEED)
        etype = {tuple(sorted((u, v))): _edge_type_func() for u, v in graph.edges()}
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            edge_type_space=EDGE_TYPE_SPACE,
            edge_type=etype,
        )
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            edge_type_init=EDGE_TYPE_INIT,
            solver_type='explicit',
            step_control='adaptive',
        )
        assert sol.shape[1] == 20
        assert_lfe_valid(time, sol, idx_to_state, STATE_SPACE)

    def test_global_interaction_lfe_runs(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            global_interaction=True,
        )
        time, sol, idx_to_state = simulate_markov_lfe(
            ips=ips,
            initial_conditions=INITIAL_CONDITIONS_FRAC,
            max_time=MAX_TIME,
            num_grid_points=20,
            solver_type='explicit',
            step_control='adaptive',
        )
        assert sol.shape[1] == 20
        assert_lfe_valid(time, sol, idx_to_state, STATE_SPACE)


# ---------------------------------------------------------------------------
# Graph utility tests
# ---------------------------------------------------------------------------

class TestGraphUtilities:

    def test_sample_graph_degree_distribution(self):
        graph = _make_graph()
        assert graph.number_of_nodes() == NUM_PARTICLES
        # All nodes should have degree ~3 in a 3-regular configuration model
        degrees = dict(graph.degree())
        avg_degree = np.mean(list(degrees.values()))
        assert avg_degree == pytest.approx(3.0, abs=0.5)

    def test_get_neighborhood(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        sim_init = _make_sim_init(graph)
        for node in list(graph.nodes())[:5]:
            nb = ips.get_neighborhood(node, sim_init)
            assert nb[0] == sim_init[node]         # root is the node's own state
            assert len(nb) == 1 + graph.degree(node)  # root + one per neighbour

    def test_compute_global_empirical_measure_sums_to_one(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            global_interaction=True,
        )
        sim_init = _make_sim_init(graph)
        meas = ips.compute_global_empirical_measure(sim_init)
        total = sum(meas.values())
        # The configuration model may remove self-loops, changing some node degrees,
        # so a small fraction of neighborhoods can be silently skipped.
        # We verify the measure is close to 1 but allow for this discrepancy.
        assert 0.9 <= total <= 1.0 + 1e-9

    def test_compute_global_empirical_measure_none_without_flag(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
            global_interaction=False,
        )
        sim_init = _make_sim_init(graph)
        meas = ips.compute_global_empirical_measure(sim_init)
        assert meas is None

    def test_renew_graph_preserves_node_count(self):
        graph = _make_graph()
        ips = GeneralizedSIR(
            state_space=STATE_SPACE, graph=graph, name='test',
            b=B, rho=RHO, deg_dist=DEG_DIST,
        )
        ips.renew_graph(seed=99)
        assert ips.num_particles == NUM_PARTICLES
