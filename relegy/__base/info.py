info_dict = {
    "DeepWalk": """
    The DeepWalk method generates the embedding based on random walks on the embedded graph. 
    Firstly, the random walks are generated. Then, a Word2Vec Skipgram model is initialized with the given parameters.
    The model is trained on the random walks to generate the embedding.
    The details may be found in: \n
    'B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning of social representations. In KDD, 2014'
    """,

    "DNGR": """The DNGR method uses a Stacked Denoising Autoencoder to generate the embedding. Firstly, a PPMI matrix of the graph is generated.
    Then the neural network is initizalized and trained with the PPMI matrix to generate the embedding.
    The details may be found in: \n
    'S. Cao, W. Lu, and Q. Xu. Deep neural networks for learning graph representations. In AAAI, 2016.'""",

    "GCN": """The Graph Convolutional Network model is used to directly classify vertices of the graph given the labels. 
    It works as a generalization of a Convolutional Network, where each layer is additionaly multiplied by an A_hat matrix obtained
    through a normalization trick.
    The details may be found in: \n
    'T.N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2016.'""",

    "GNN": """Graph Neural Network model is used to directly classify vertices of the graph given the labels by generating vertex state representation. 
    To generate a vertex state it uses the fixed-point theorem on a non-linear constricting transformation, which is represented by 
    an optimizable neural network. Given the representation, it performs another non-linear transformation on state representation 
    to classify vertices.
    The details may be found in: \n
    'Scarselli, F., Gori, M., Tsoi, A., Hagenbuchner, M. & Monfardini, G. 2009, 'The graph neural network model', IEEE Transactions on
Neural Networks, vol. 20, no. 1, pp. 61-80.'""",

    "GraphFactorization": """The Graph Factorization method aims to minimze a loss function with an assumption that the adjacency matrix 
    should be approximated by ZZ^T, where Z is the embedding matrix. 
    The details may be found in: \n
    'A. Ahmed, N. Shervashidze,
    S. Narayanamurthy, V. Josifovski, and A.J. Smola. Distributed large-scale natural graph factorization. In WWW,
    2013'""",

    "GraphWave": """The GraphWave method generates spectral graph wavelets associated with the graph to obtain a 
    diffusion patter for every node for given kernel. Then the wavelet coefficients are treated as a probability distribution to obtain empirical characteristic functions.
    The GraphWave method captures structural similarity of the nodes.
    The details may be found in: \n
    'C. Donnat, M. Zitnik, D. Hallac, and J. Leskovec. Learning structural node embeddings via diffusion wavelets. arXiv
preprint arXiv:1710.10321, 2017.'""",

    "GraRep": """The GraRep method obtains the embedding from Singular Value Decompositions of denoised powers of the adjacency matrix of the graph.
    The details may be found in: \n
    'S. Cao, W. Lu, and Q. Xu. Grarep: Learning graph representations with global structural information. In KDD, 2015'""",

    "HARP": """The HARP method improves DeepWalk and Node2Vec methods by generating a family of graphs and training the network through the family instead of a single graph.
    Word2Vec models are then iteratively initialized with previously acquired weights. 
    The details may be found in: \n
    'H. Chen, B. Perozzi, Y. Hu, and S. Skiena. Harp: Hierarchical representation learning for networks. arXiv preprint
arXiv:1706.07845, 2017.'""",

    "HOPE": """The HOPE method uses a Singular Value Decomposition of a similarity matrix in order to obtain the embedding.
    The details may be found in: \n
    'M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu. Asymmetric transitivity preserving graph embedding. In KDD, 2016.'""",

    "LaplacianEigenmaps": """The Laplacian Eigenmaps method solves a constrained optimization problem in order to map the vertices to points in a space
    so that two vertices are close when they are connected.
    The details may be found in: \n
    'M. Belkin and P. Niyogi. Laplacian eigenmaps and spectral techniques for embedding and clustering. In NIPS, 2002.'""",

    "LINE": """The LINE method minimizes two loss functions to capture local node similarity in the graph.
    The details may be found in: \n
    ' J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In
    WWW, 2015.'""",

    "Node2Vec": """The Node2Vec uses random walks in a similar manner to the DeepWalk method, however, the random walks are now biased in order
    to better capture local similarity of the nodes.
    The details may be found in: \n
    'A. Grover and J. Leskovec. node2vec: Scalable feature learning for networks. In KDD, 2016'""",

    "SDNE": """The SDNE method used an autoencoder network to embed the graph's adjacency matrix in a lower dimension. 
    The details may be found in: \n
    'D. Wang, P. Cui, and W. Zhu. Structural deep network embedding. In KDD, 2016.'""",

    "Struc2Vec": """The Struc2Vec method captures structural similarity of the nodes by generating random walks on a new graph, where structurally similar vertices are close to each other.
    Then the random walks are used to train a Word2Vec Skipgram model.
    The details may be found in: \n
    'L.F.R. Ribeiro, P.H.P. Saverese, and D.R. Figueiredo. struc2vec: Learning node representations from structural
identity. In KDD, 2017.'
    """
}