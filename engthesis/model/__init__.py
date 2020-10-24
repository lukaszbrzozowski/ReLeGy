from abc import ABC, abstractmethod

from networkx import Graph
from numpy import ndarray


class Model(ABC):

    def __init__(self, graph):
        self.__graph: Graph = graph
        super().__init__()

    @abstractmethod
    def embed(self) -> ndarray: pass

    @abstractmethod
    def info(self) -> str: pass

    def get_graph(self) -> Graph: return self.__graph
