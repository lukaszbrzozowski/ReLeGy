from networkx import adjacency_matrix

from engthesis.model.base import Model
import numpy as np

class LINE(Model):

    def __init__(self, graph, **kwargs):
        super().__init__(graph.to_directed())
        self.__A = kwargs["A"] if "A" in kwargs else adjacency_matrix(self.get_graph())
        self.__d = kwargs["d"] if "d" in kwargs else 2
        self.__alpha1 = kwargs["alpha1"] if "alpha1" in kwargs else 1e-4
        self.__alpha2 = kwargs["alpha2"] if "alpha2" in kwargs else 1e-4
        self.__epochs = kwargs["epochs"] if "epochs" in kwargs else 400
        self.__batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 30
        self.__lmbd1 = kwargs["lmbd1"] if "lmbd1" in kwargs else 1e-1
        self.__lmbd2 = kwargs["lmbd2"] if "lmbd2" in kwargs else 1e-1
        self.__U1 = kwargs["U1"] if "U1" in kwargs else np.random.random((len(graph.nodes),self.__d))
        self.__U2 = kwargs["U2"] if "U2" in kwargs else np.random.random((len(graph.nodes),self.__d))
        self.__model = None

    def embed(self):
        Frob = lambda U: np.sum(U**2)
        graph = self.get_graph()
        n_edges = len(graph.edges)
        p1 = lambda x, y: 1/(1+np.exp(-np.dot(x,y)))
        p2 = lambda x, y: np.exp(np.dot(x,y))/(np.sum(self.__U2 @ y))
        o1 = lambda G: -sum([self.__A[i, j]*np.log(p1(self.__U1[i,:], self.__U1[j,:])) for (i,j) in G.edges])
        o2 = lambda G: -sum([self.__A[i, j]*np.log(p2(self.__U2[i,:], self.__U2[j,:])) for (i,j) in G.edges])
        grad1 = lambda i, j: self.__A[i, j]*(-self.__U1[j,:])*np.exp(-np.dot(self.__U1[i,:],self.__U1[j,:]))/(1+np.exp(-np.dot(self.__U1[i,:],self.__U1[j,:])))**2
        grad2 = lambda i, j: self.__A[i, j]*(self.__U2[j,:]*np.exp(np.dot(self.__U2[i,:],self.__U2[j,:]))*np.sum(self.__U2 @ self.__U2[j,:]) - np.exp(np.dot(self.__U2[i,:],self.__U2[j,:]))*self.__U2[j,:])/(np.sum(self.__U2 @ self.__U2[j,:]))**2
        for iter in range(self.__epochs):
            resampled_edges = np.copy(graph.edges)
            np.random.shuffle(resampled_edges)
            mini_batches = [resampled_edges[i*self.__batch_size:min(n_edges,i+1*self.__batch_size)] for i in range(n_edges//self.__batch_size)]
            for batch in mini_batches:
                step1 = np.zeros(self.__U1.shape)
                step2 = np.zeros(self.__U2.shape)
                for (i, j) in batch:
                    step1[i,:] += grad1(i,j) + self.__lmbd1*self.__U1[i,:]
                    step2[i,:] += grad2(i,j) + self.__lmbd2*self.__U2[i,:]
                self.__U1 -= self.__alpha1*step1
                self.__U2 -= self.__alpha2*step2
            print(f"Epoch: {iter+1}, 1st-order objective: {o1(graph):.2f}, 1st-order Frob:{Frob(self.__U1):.2f} 2nd-order objective: {o2(graph):.2f}, 2nd-order Frob:{Frob(self.__U2):.2f}", end="\r")
        self.__model = np.c_[self.__U1, self.__U2]
        return self.__model

    def info(self):
        pass