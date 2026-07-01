from sparseips.ips_class import ParticleSystem, MeanFieldParticleSystem
from sparseips.jax_mlfe import compute_jax_static_args
from sparseips.markov_lfe import simulate_markov_lfe
from sparseips.mean_field import simulate_mean_field
from sparseips.util import pad_indices
import jax
import optax
import abc
import jax.flatten_util
import jax.numpy as jnp
import numpy as np


class Optimizer(abc.ABC):
    """
    Abstract base class for JAX-based optimization strategies.
    Any new optimizer must inherit from this and implement the two abstract methods.
    """

    @abc.abstractmethod
    def init(self, initial_params):
        """
        Initialize the optimizer state.
        Args:
            initial_params: The initial parameters (usually flattened).
        Returns:
            opt_state: The initial optimizer state.
        """
        pass

    @abc.abstractmethod
    def make_step_fn(self, objective_fn):
        """
        Factory method that creates a JIT-compiled step function.

        Args:
            objective_fn: A function that takes parameters and returns a scalar loss.
                          (This function is 'baked' into the returned step function).

        Returns:
            step_fn: A function with signature:
                     (params_sparse, state, key) -> (new_params, new_state, loss)
        """
        pass


class EvolutionStrategy(Optimizer):
    """
    Implements the OpenAI Evolution Strategy (ES).
    """

    def __init__(self, num_params, population_size=64, sigma=0.1, step_size=0.05):
        self.num_params = num_params
        self.pop_size = population_size
        self.sigma = sigma
        self.optimizer = optax.adam(learning_rate=step_size)

    def init(self, initial_params):
        """Initializes the optimizer state."""
        return self.optimizer.init(initial_params)

    def make_step_fn(self, objective_fn):
        """
        Returns a JIT-compiled step function specific to the given objective.
        We do this here so 'objective_fn' is baked into the compiled kernel.
        """
        batch_objective = jax.vmap(objective_fn)

        @jax.jit
        def step(flat_params, state, key):
            # A. Noise
            noise = jax.random.normal(key, shape=(self.pop_size, self.num_params))

            # B. Candidates (Antithetic)
            pos = flat_params + self.sigma * noise
            neg = flat_params - self.sigma * noise
            candidates = jnp.concatenate([pos, neg])

            # C. Evaluate
            all_costs = batch_objective(candidates)

            # D. Estimate Gradient
            pos_costs = all_costs[:self.pop_size]
            neg_costs = all_costs[self.pop_size:]

            weighted_noise = noise.T @ (pos_costs - neg_costs)
            grad_est = weighted_noise / (2 * self.pop_size * self.sigma)

            # E. Update
            updates, new_state = self.optimizer.update(grad_est, state, flat_params)
            new_params = optax.apply_updates(flat_params, updates)

            return new_params, new_state, jnp.mean(all_costs)

        return step


def optimize_policy(ips,
                    ode_params,
                    objective_fn,
                    initial_params,
                    constraint_fn=None,
                    optimizer: Optimizer = None,
                    num_iterations=100,
                    verbose=False,
                    save_dir=None):

    default_ode_params = {
        'initial_conditions': None,
        'max_time': 5.0,
        'num_grid_points': 100,
        'solver_type': 'explicit',
        'step_control': 'adaptive',
        'verbose': False,
    }

    ode_params = default_ode_params | ode_params

    flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(initial_params)

    # prepare for sparse particle simulation
    if isinstance(ips, ParticleSystem):
        # precompute indices
        ode_state_to_index, mlfe_static_args, mlfe_sparse_indices = compute_jax_static_args(ips)

        # define simulator
        def objective_fn_wrapper(params):
            unflat_param = unflatten_fn(params)
            if constraint_fn is not None:
                unflat_param = constraint_fn(unflat_param)

            ips.set_params(**unflat_param)
            time, sol, _ = simulate_markov_lfe(
                ips=ips,
                **ode_params,
                static_args=mlfe_static_args,
                sparse_indices=mlfe_sparse_indices
            )

            return objective_fn(time, sol, unflat_param)

    # prepare for mean-field systems
    elif isinstance(ips, MeanFieldParticleSystem):
        def objective_fn_wrapper(params):
            unflat_param = unflatten_fn(params)
            if constraint_fn is not None:
                unflat_param = constraint_fn(unflat_param)

            ips.set_params(**unflat_param)
            time, sol, _ = simulate_mean_field(
                mfps=ips,
                **ode_params
            )

            return objective_fn(time, sol, unflat_param)

    else:
        raise ValueError("Unsupported IPS type for optimization.")

    step_fn = optimizer.make_step_fn(objective_fn_wrapper)
    opt_state = optimizer.init(flat_params)

    # 5. Run Loop
    key = jax.random.PRNGKey(42)
    losses = []

    print(f"Starting Optimization...")

    for i in range(num_iterations):
        iter_key, key = jax.random.split(key)
        flat_params, opt_state, loss = step_fn(flat_params, opt_state, iter_key)

        losses.append(float(loss))

        if verbose and i % 5 == 0:
            print(f"Iter {i}: Loss = {loss:.4f}")
            np.savez(save_dir + f'/{i}.npz', **unflatten_fn(flat_params))

    final_params = constraint_fn(unflatten_fn(flat_params)) if constraint_fn is not None else unflatten_fn(flat_params)
    np.savez(save_dir + f'/params_final.npz', **final_params)
    return final_params, losses
