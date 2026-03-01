from abc import ABC, abstractmethod
from dataclasses import dataclass

class JumpRate(ABC):
    @abstractmethod
    def __call__(
        self,
        src: any,
        tgt: any,
        neighbors: tuple[any],
        neighbors_vertex_type: tuple[any] = None,
        neighbors_edge_type: tuple[any] = None,
        neighbors_edge_state: tuple[any] = None,
        meas: dict[tuple[any], float] = None,
    ) -> float:
        """Compute the transition rate of a vertex from src to tgt given its neighbors."""
        pass


class EdgeJumpRate(ABC):
    @abstractmethod
    def __call__(
        self,
        src: any,
        tgt: any,
        neighbors: (any, any),
        meas: dict[tuple[any], float] = None,
    ) -> float:
        """Compute the transition rate of an edge from src to tgt given two connected vertices."""
        pass


@dataclass
class InteractionContexts:
    vertex_types: tuple[any] = None
    edge_types: tuple[any] = None
    edge_states: tuple[any] = None