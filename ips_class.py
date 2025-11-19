import networkx as nx
import numpy as np
from abc import abstractmethod
from itertools import product
from collections import Counter


class ParticleSystem:
    def __init__(
        self,
        state_space: list[any],
        graph: nx.Graph,
        name: str = None,
        deg_dist: dict[int, float] = None,
        vertex_type_space: list[any] = None,
        vertex_type: dict[int, any] = None,
        edge_type_space: list[any] = None,
        edge_type: dict[tuple[int, int], any] = None,
        edge_state: list[any] = None,
        edge_rate: callable = None,
        global_interaction: bool = False,
    ):

        self.name = name
        self.num_particles = graph.number_of_nodes()
        self.state_space = state_space
        self.graph = graph
        self.deg_dist = deg_dist
        if self.deg_dist is None:
            self.deg_dist = self.get_empirical_degree_distribution()
        deg_supp = [i for (i, p) in self.deg_dist.items() if p > 0]
        self.neighborhood_state_space = [
            (root,) + children
            for k in deg_supp
            for (root, children) in product(
                self.state_space, product(self.state_space, repeat=k)
            )
        ]

        self.vertex_type_space = vertex_type_space
        self.vertex_type = vertex_type
        self.edge_type_space = edge_type_space
        self.edge_type = edge_type
        self.edge_state = edge_state
        self.edge_rate = edge_rate
        self.global_interaction = global_interaction

    @abstractmethod
    def rate(
        self,
        src: any,
        tgt: any,
        neighbors: tuple[any],
        neighbors_vertex_type: list[any] = None,
        neighbors_edge_type: list[any] = None,
        meas: dict[tuple[any], float] = None,
    ) -> float:
        """
        Compute the rate of transition from source_state to target_state for a given node.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_empirical_degree_distribution(self):
        if self.deg_dist is not None:
            return self.deg_dist

        # from the graph, compute the maximum degree, degree distribution, and support of the degree distribution
        max_deg = max(dict(self.graph.degree()).values())
        deg_dist = {d: 0 for d in range(max_deg + 1)}
        for _, d in self.graph.degree():
            deg_dist[d] += 1 / self.num_particles

        # set field in the class
        self.deg_dist = deg_dist

        return deg_dist

    @staticmethod
    def sample_graph_from_deg_dist(deg_dist, num_particles, seed):
        # set seed
        np.random.seed(seed)
        # construct degree sequence by sampling iid from degree distribution
        deg_seq = [
            np.random.choice(list(deg_dist.keys()), p=list(deg_dist.values()))
            for _ in range(num_particles)
        ]
        # check if sum of degree sequence is even, if not, make it even
        if sum(deg_seq) % 2 != 0:
            deg_seq[0] += 1

        # draw configuration model with specified degree distribution
        new_graph = nx.configuration_model(deg_seq, seed=seed)
        new_graph = nx.Graph(new_graph)  # convert to simple graph
        new_graph.remove_edges_from(nx.selfloop_edges(new_graph))  # remove self-loops

        return new_graph

    def renew_graph(
        self,
        seed: int,
        vertex_type_func: callable = None,
        edge_type_func: callable = None,
        edge_state_func: callable = None,
        edge_rate_func: callable = None,
    ):
        # sample a new graph according to deg distribution

        if self.deg_dist is None:
            self.get_empirical_degree_distribution()

        self.graph = ParticleSystem.sample_graph_from_deg_dist(
            self.deg_dist, self.num_particles, seed
        )

        # renew vertex type
        if self.vertex_type is not None:
            self.vertex_type = {node: vertex_type_func() for node in self.graph.nodes()}
        # renew edge type
        if self.edge_type is not None:
            self.edge_type = {
                tuple(sorted((u, v))): edge_type_func() for u, v in self.graph.edges()
            }

        return self

    def sim_rate(
        self,
        node: int,
        source_state: any,
        target_state: any,
        current_config: dict[int, any],
    ):
        # get neighbors of the source state
        neighbors_state = tuple(
            [current_config[neighbor] for neighbor in self.graph.neighbors(node)]
        )
        # get neighbors vertex type
        neighbors_vertex_type = (
            None
            if self.vertex_type is None
            else [self.vertex_type[node]]
            + [self.vertex_type[neighbor] for neighbor in self.graph.neighbors(node)]
        )
        # get neighbors edge type
        neighbors_edge_type = (
            None
            if self.edge_type is None
            else [
                self.edge_type[tuple(sorted((node, neighbor)))]
                for neighbor in self.graph.neighbors(node)
            ]
        )
        # get global neighborhood empirical measure in the form of dictionary
        if self.global_interaction:
            global_empirical_measure = {
                neighborhood: 0 for neighborhood in self.neighborhood_state_space
            }
            for vertex in range(self.num_particles):
                neighborhood = (current_config[vertex],) + tuple(
                    current_config[neighbor]
                    for neighbor in self.graph.neighbors(vertex)
                )
                try:
                    global_empirical_measure[neighborhood] += 1 / self.num_particles
                except KeyError:
                    pass
        else:
            global_empirical_measure = None

        return self.rate(
            source_state,
            target_state,
            neighbors_state,
            neighbors_vertex_type=neighbors_vertex_type,
            neighbors_edge_type=neighbors_edge_type,
            meas=global_empirical_measure,
        )

    def get_state_to_index_map(self):
        """
        Get a mapping from state to index for the state space.
        :return: A dictionary mapping each state to its index.
        """
        return {state: i for i, state in enumerate(self.state_space)}


class MeanFieldParticleSystem:
    def __init__(
        self,
        state_space: list[any],
        num_particles: int,
        name: str = None,
    ):
        self.state_space = state_space
        self.num_particles = num_particles
        self.name = name

    @abstractmethod
    def rate(self, src: any, tgt: any, meas: dict[tuple[any], float]) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_empirical_measure(
        self, current_state: dict[int, any]
    ) -> dict[any, float]:
        """
        Compute the empirical measure (fraction of particles in each state).

        Parameters:
        -----------
        current_state : Dict[int, Any]
            Current state of all particles

        Returns:
        --------
        Dict[Any, float]
            Dictionary mapping each state to its fraction in the population
        """
        state_counts = Counter(current_state.values())
        return {
            state: count / self.num_particles for state, count in state_counts.items()
        }

    def sim_rate(self, src: any, tgt: any, meas: dict[any, float]):
        """
        Simulate the rate of transition from source_state to target_state for a given node.
        :param source_state: the state particle is jumping from
        :param target_state: the state particle is jumping to
        :param empirical_measure: a dictionary of states to its fraction in the population
        :return:
        """
        return self.rate(src, tgt, meas)
