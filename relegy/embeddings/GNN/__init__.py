import tensorflow as tf
import numpy as np
import datetime as time
import networkx as nx
import scipy.sparse as sp
from networkx import Graph

from relegy.__base import Model

construct_verification = {"graph": [(lambda x: issubclass(type(x), Graph), "'graph' must be a networkx graph")]}


# class for the core of the architecture
class GNN(Model):
    """
    The Graph Neural Network method implementation. \n
    The details may be found in: \n
    'Scarselli, F., Gori, M., Tsoi, A., Hagenbuchner, M. & Monfardini, G. 2009, 'The graph neural network model', IEEE Transactions on
Neural Networks, vol. 20, no. 1, pp. 61-80.'
    """

    @Model._verify_parameters(rules_dict=construct_verification)
    def __init__(self, graph: nx.Graph):
        """
        GNN -- constructor (step I)

        @param graph: The graph to be embedded. Nodes of the graph must be a sorted array from 0 to n-1, where n is
        the number of vertices.
        create GNN instance. Feed this parameters:
        :net:  Net instance - it contains state network, output network, initialized weights, loss function and metric;
        :input_dim: dimension of the input
        :output_dim: dimension of the output
        :state_dim: dimension for the state
        :max_it:  maximum number of iteration of the state convergence procedure
        :optimizer:  optimizer instance
        :learning_rate: learning rate value
        :threshold:  value to establish the state convergence
        :graph_based: flag to denote a graph based problem
        :param: name of the experiment
        :config: ConfigProto protocol buffer object, to set configuration options for a session
        """
        super().__init__(graph)
        self.inp = None
        self.arcnode = None
        self.graphnode = None
        self.labels = None
        self.mask_test = None
        self.mask_train = None
        self.input_dim = None
        self.output_dim = None
        self.state_dim = None
        self.max_iter = None
        self.num_epoch = None
        self.net = None
        self.optimizer = None
        self.state_threshold = None
        self.graph_based = None
        self.mask_flag = None

        self.input_dim = None
        self.state_dim = None
        self.output_dim = None
        self.state_input = None
        self.state_l1 = None
        self.output_l1 = None
        self.modelSt = None
        self.modelOut = None
        self.state = None
        self.state_old = None

    @Model._init_in_init_model_fit
    def initialize(self, idx_labels):
        """
        GNN - Initialize (step II) \n
        Generates internal graph representation and transforms labels to suitable format.

        @param idx_labels: ndarray nx2 of pairs (node_id, label)
        """
        inp, arcnode, graphnode, labels = self.__from_nx_to_GNN(self.get_graph(), idx_labels)

        self.inp = inp
        self.arcnode = arcnode
        self.graphnode = graphnode
        self.labels = labels

    @Model._init_model_in_init_model_fit
    def initialize_model(self, embed_dim=4,
                         max_it=50, optimizer=tf.keras.optimizers.Adam, learning_rate=0.01, threshold=0.01,
                         mask_flag=True, mask_test=None, mask_train=None):
        """
        GNN - initialize_model (step III) \n
        Initializes the Graph Neural Network model.

        @param embed_dim: dimension for the embedding.
        @param max_it: maximum number of iteration of the state convergence procedure.
        @param optimizer: optimizer instance.
        @param learning_rate: learning rate value.
        @param threshold: value to establish the state convergence.
        @param mask_flag: flag to denote using masks for training/test.
        @param mask_test: testing masks.
        @param mask_train: training masks.
        """
        self.input_dim = self.inp.shape[1]
        self.output_dim = self.labels.shape[1]
        self.state_dim = embed_dim
        self.state_input = self.input_dim - 2 + self.state_dim
        self.state_l1 = 2*self.state_dim
        self.output_l1 = 2*self.state_dim
        self.modelSt = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.state_input),
            tf.keras.layers.Dense(self.state_l1, activation=tf.nn.tanh),
            tf.keras.layers.Dense(self.state_dim, activation=tf.nn.tanh)
        ])
        self.modelOut = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.state_dim),
            tf.keras.layers.Dense(self.output_l1, activation=tf.nn.tanh),
            tf.keras.layers.Dense(self.output_dim, activation=tf.nn.softmax)
        ])

        self.max_iter = max_it
        self.optimizer = optimizer(learning_rate, name="optim")
        self.state_threshold = threshold
        self.mask_flag = mask_flag

        n = len(self.get_graph().nodes)

        if mask_test is None:
            self.mask_test = np.zeros(shape=(n,), dtype=np.float32())
            self.mask_test[np.random.choice(np.arange(n), size=n // 10, replace=False)] = 1
        else:
            self.mask_test = mask_test

        if mask_train is None:
            self.mask_train = 1.0 - self.mask_test
        else:
            self.mask_train = mask_train

    @Model._fit_in_init_model_fit
    def fit(self, num_epoch=100):
        """
        GCN - fit (step IV) \n
        Trains the GNN model.

        @param num_epoch: Number of training epochs.
        """
        count = 0

        ######

        for j in range(0, num_epoch):
            with tf.GradientTape() as tape:
                out, num, st = self.__loop()
                variables = self.modelSt.trainable_variables + self.modelOut.trainable_variables

                # loss
                if self.mask_flag:
                    loss = self.__loss(out, self.labels, mask=self.mask_train)
                    loss_test = self.__loss(out, self.labels, mask=self.mask_train)
                    metric = self.__metric(self.labels, out, mask=self.mask_test)
                    metric_test = self.__metric(self.labels, out, mask=self.mask_test)
                else:
                    loss = self.__loss(out, self.labels)
                    metric = self.__metric(self.labels, out)

                grads = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(grads, variables), name='train_op')
                # _, it = self.__train(inputs=self.inp, ArcNode=self.arcnode, target=self.labels, step=count,
                #                      mask=self.mask_train)

            if count % 10 == 0:
                print("Epoch ", count)
                if self.mask_flag:
                    print("Training: ", loss, num, metric)
                    print("Test: ", loss_test, num, metric_test)
                else:
                    print("Results: ", loss, num, metric)

            count = count + 1

    @Model._embed_in_init_model_fit
    def embed(self):
        _, _, result = self.__loop()
        return result.numpy()

    @staticmethod
    def fast_embed(graph: Graph, idx_labels, embed_dim=4,
                   max_it=50, optimizer=tf.keras.optimizers.Adam, learning_rate=0.01, threshold=0.01,
                   mask_flag=False, mask_train=None, mask_test=None, num_epoch=100):
        gnn = GNN(graph)
        gnn.initialize(idx_labels)
        gnn.initialize_model(embed_dim, max_it, optimizer, learning_rate, threshold,
                             mask_flag, mask_train, mask_test)
        gnn.fit(num_epoch)
        return gnn.embed()

    def __from_EN_to_GNN(self, E, N):
        """
        :param E: # E matrix - matrix of edges : [[id_p, id_c, graph_id],...]
        :param N: # N matrix - [node_features, graph_id (to which the node belongs)]
        :return: # L matrix - list of graph targets [tar_g_1, tar_g_2, ...]
        """
        N_full = N
        N = N[:, :-1]  # avoid graph_id
        e = E[:, :2]  # take only first tow columns => id_p, id_c
        feat_temp = np.take(N, e, axis=0)  # take id_p and id_c  => (n_archs, 2, label_dim)
        feat = np.reshape(feat_temp, [len(E), -1])  # (n_archs, 2*label_dim) => [[label_p, label_c], ...]
        # creating input for gnn => [id_p, id_c, label_p, label_c]
        inp = np.concatenate((E[:, :2], feat), axis=1)
        # creating arcnode matrix, but transposed
        """
        1 1 0 0 0 0 0 
        0 0 1 1 0 0 0
        0 0 0 0 1 1 1    
        """  # for the indices where to insert the ones, stack the id_p and the column id (single 1 for column)
        arcnode = tf.SparseTensor(indices=np.stack((E[:, 0], np.arange(len(E))), axis=1),
                                  values=np.ones([len(E)]).astype(np.float32),
                                  dense_shape=[len(N), len(E)])

        # get the number of graphs => from the graph_id
        num_graphs = int(max(N_full[:, -1]) + 1)
        # get all graph_ids
        g_ids = N_full[:, -1]
        g_ids = g_ids.astype(np.int32)

        # creating graphnode matrix => create identity matrix get row corresponding to id of the graph
        # graphnode = np.take(np.eye(num_graphs), g_ids, axis=0).T
        # substitued with same code as before
        graphnode = tf.SparseTensor(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
                                    values=np.ones([len(g_ids)]).astype(np.float32),
                                    dense_shape=[num_graphs, len(N)])

        # print(graphnode.shape)

        return inp, arcnode, graphnode

    def __from_nx_to_GNN(self, graph: nx.Graph, idx_labels: np.ndarray):

        edges = np.array(list(graph.edges))  # node indices from 0 to |G|-1
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]  # reorder list of edges also by second column
        features = sp.eye(np.max(edges + 1), dtype=np.float32).tocsr()

        labels = np.eye(max(idx_labels[:, 1]) + 1, dtype=np.int32)[idx_labels[:, 1]]  # one-hot encoding of labels

        E = np.concatenate((edges, np.zeros((len(edges), 1), dtype=np.int32)), axis=1)
        N = np.concatenate((features.toarray(), np.zeros((features.shape[0], 1), dtype=np.int32)), axis=1)

        inp, arcnode, graphnode = self.__from_EN_to_GNN(E, N)
        return inp, arcnode, graphnode, labels

    def __loss(self, output, target, mask=None):
        # method to define the loss function
        # lo = tf.losses.softmax_cross_entropy(target, output)
        if mask is None:
            mask = np.ones(target.shape[0])
        output = tf.maximum(output, 1e-7, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.math.log(output), 1)

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        xent *= mask
        lo = tf.reduce_mean(xent)

        return lo

    def __metric(self, target, output, mask=None):
        # method to define the evaluation metric
        if mask is None:
            mask = np.ones(target.shape[0])

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)

    def __convergence(self, a, state, old_state, k):
        # body of the while cycle used to iteratively calculate state

        # assign current state to old state
        old_state = state

        # grub states of neighboring node
        gat = tf.gather(old_state, tf.cast(a[:, 1], tf.int32))

        # slice to consider only label of the node and that of it's neighbor
        # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
        # equivalent code
        sl = a[:, 2:]
        x = tf.concat([sl, gat], axis=1)

        # concat with retrieved state
        # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
        res = self.modelSt(x, training=True)
        state = tf.sparse.sparse_dense_matmul(self.arcnode, res)

        # update the iteration counter
        k = k + 1

        return a, state, old_state, k

    def __condition(self, a, state, old_state, k):
        # evaluate condition on the convergence of the state
        # evaluate distance by state(t) and state(t-1)

        outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
        # vector showing item converged or not (given a certain threshold)
        checkDistanceVec = tf.greater(outDistance, self.state_threshold)

        c1 = tf.reduce_any(checkDistanceVec)
        c2 = tf.less(self.k, self.max_iter)

        return tf.logical_and(c1, c2)

    def __loop(self):
        # call to loop for the state computation and compute the output
        # compute state
        self.k = 0
        self.state = np.zeros((self.arcnode.dense_shape[0], self.state_dim))
        self.state_old = np.ones((self.arcnode.dense_shape[0], self.state_dim))

        res, st, old_st, num = tf.while_loop(self.__condition, self.__convergence,
                                             [self.inp, self.state, self.state_old, self.k])

        out = self.modelOut(st)
        return out, num, st