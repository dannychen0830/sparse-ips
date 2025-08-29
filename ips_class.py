import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Callable, Any
from abc import abstractmethod
from itertools import product
from collections import Counter


class ParticleSystem:
    def __init__(self,
                 state_space: List[Any],
                 graph: nx.Graph,
                 name: str = None,
                 deg_dist: Dict[int, float] = None,
                 vertex_type_space: List[Any] = None,
                 vertex_type: Dict[int, Any] = None,
                 edge_type_space: List[Any] = None,
                 edge_type: Dict[Tuple[int, int], Any] = None,
                 edge_state: List[Any] = None,
                 edge_rate: Callable = None,
                 global_interaction: bool = False):

        self.name = name
        self.num_particles = graph.number_of_nodes()
        self.state_space = state_space
        self.graph = graph
        self.deg_dist = deg_dist
        if deg_dist is None:
            deg_supp = None
        else:
            deg_supp = [i for (i, p) in deg_dist.items() if p > 0]
        self.neighborhood_state_space = [(root,) + children for k in deg_supp for (root, children) in
                              product(self.state_space, product(self.state_space, repeat=k))]

        self.vertex_type_space = vertex_type_space
        self.vertex_type = vertex_type
        self.edge_type_space = edge_type_space
        self.edge_type = edge_type
        self.edge_state = edge_state
        self.edge_rate = edge_rate
        self.global_interaction = global_interaction

    @abstractmethod
    def rate(self, source_state: Any,
             target_state: Any,
             neighbors_state: Tuple[Any],
             neighbors_vertex_type: List[Any] = None,
             neighbors_edge_type: List[Any] = None,
             global_empirical_measure: Dict[Tuple[Any], float] = None) -> float:
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

    def renew_graph(self, seed: int, vertex_type_func: Callable = None, edge_type_func: Callable = None, edge_state_func: Callable = None, edge_rate_func: Callable = None):
        # sample a new graph according to deg distribution

        if self.deg_dist is None:
            self.get_empirical_degree_distribution()

        # set seed
        np.random.seed(seed)
        # construct degree sequence by sampling iid from degree distribution
        deg_seq = [np.random.choice(list(self.deg_dist.keys()), p=list(self.deg_dist.values())) for _ in range(self.num_particles)]

        # draw configuration model with specified degree distribution
        new_graph = nx.configuration_model(deg_seq, seed=seed)
        new_graph = nx.Graph(new_graph)  # convert to simple graph
        new_graph.remove_edges_from(nx.selfloop_edges(new_graph))  # remove self-loops

        self.graph = new_graph

        # renew vertex type
        if self.vertex_type is not None:
            self.vertex_type = {node: vertex_type_func() for node in self.graph.nodes()}
        # renew edge type
        if self.edge_type is not None:
            self.edge_type = {(u, v): edge_type_func() for u, v in self.graph.edges()}

        return self

    def sim_rate(self, node: int, source_state: Any, target_state: Any, current_config: Dict[int, Any]):
        # get neighbors of the source state
        neighbors_state = tuple([current_config[neighbor] for neighbor in self.graph.neighbors(node)])
        # get neighbors vertex type
        neighbors_vertex_type = None \
            if self.vertex_type is None \
            else [self.vertex_type[node]] + [self.vertex_type[neighbor] for neighbor in self.graph.neighbors(node)]
        # get neighbors edge type
        neighbors_edge_type = None \
            if self.edge_type is None \
            else [self.edge_type[tuple(sorted((node, neighbor)))] for neighbor in self.graph.neighbors(node)]
        # get global neighborhood empirical measure in the form of dictionary
        if self.global_interaction:
            global_empirical_measure = {neighborhood: 0 for neighborhood in self.neighborhood_state_space}
            for vertex in range(self.num_particles):
                neighborhood = (current_config[vertex],) + tuple(current_config[neighbor] for neighbor in self.graph.neighbors(vertex))
                global_empirical_measure[neighborhood] = global_empirical_measure.get(neighborhood, 1 / self.num_particles)
        else:
            global_empirical_measure = None

        return self.rate(source_state,
                         target_state,
                         neighbors_state,
                         neighbors_vertex_type=neighbors_vertex_type,
                         neighbors_edge_type=neighbors_edge_type,
                         global_empirical_measure=global_empirical_measure)

    def get_state_to_index_map(self):
        """
        Get a mapping from state to index for the state space.
        :return: A dictionary mapping each state to its index.
        """
        return {state: i for i, state in enumerate(self.state_space)}


class MeanFieldParticleSystem():
    def __init__(self,
                 state_space: List[Any],
                 num_particles: int,
                 name: str = None):
        self.state_space = state_space
        self.num_particles = num_particles
        self.name = name

    @abstractmethod
    def rate(self, source_state: Any,
             target_state: Any,
             global_empirical_measure: Dict[Tuple[Any], float]) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_empirical_measure(self, current_state: Dict[int, Any]) -> Dict[Any, float]:
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
        return {state: count / self.num_particles for state, count in state_counts.items()}

    def sim_rate(self, source_state: Any, target_state: Any, empirical_measure: Dict[Any, float]):
        """
        Simulate the rate of transition from source_state to target_state for a given node.
        :param source_state: the state particle is jumping from
        :param target_state: the state particle is jumping to
        :param empirical_measure: a dictionary of states to its fraction in the population
        :return:
        """
        return self.rate(source_state, target_state, empirical_measure)
