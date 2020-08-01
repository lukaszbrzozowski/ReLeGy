from abc import ABC, abstractmethod
from networkx import Graph
from numpy import ndarray

class Model(ABC):

    __graph: Graph

    def __init__(self, graph):
        self.__graph = graph
        super().__init__()

    @abstractmethod
    def embed(self) -> ndarray: pass

    @abstractmethod
    def info(self) -> str: pass

    def getGraph(self) -> Graph: return self.__graph