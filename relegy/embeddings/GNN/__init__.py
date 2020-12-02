import tensorflow.compat.v1 as tf
import numpy as np
import datetime as time
import networkx as nx
import scipy.sparse as sp
from collections import namedtuple
from networkx import Graph

from relegy.__base import Model

SparseMatrix = namedtuple("SparseMatrix", "indices values dense_shape")

construct_verification = {"graph": [(lambda x: type(x) == Graph, "'graph' must be a networkx graph")]}


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
        :tensorboard:  boolean flag to activate tensorboard
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
        self.tensorboard = None
        self.max_iter = None
        self.num_epoch = None
        self.net = None
        self.optimizer = None
        self.state_threshold = None
        self.graph_based = None
        self.mask_flag = None

        self.session = None

    @Model._init_in_init_model_fit
    def initialize(self, idx_labels):
        """
        GraRep - Initialize (step II) \n
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
                         max_it=50, optimizer=tf.train.AdamOptimizer, learning_rate=0.01, threshold=0.01,
                         graph_based=False,
                         param=str(time.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), config=None, tensorboard=False,
                         mask_flag=True, mask_test=None, mask_train=None):
        """
        GNN - initialize_model (step III) \n
        Initializes the Graph Neural Network model.

        @param embed_dim: dimension for the embedding.
        @param max_it: maximum number of iteration of the state convergence procedure.
        @param optimizer: optimizer instance.
        @param learning_rate: learning rate value.
        @param threshold: value to establish the state convergence.
        @param graph_based: flag to denote a graph based problem.
        @param param: name of the experiment.
        @param config: ConfigProto protocol buffer object, to set configuration options for a session.
        @param tensorboard: boolean flag to activate tensorboard.
        @param mask_flag: flag to denote using masks for training/test.
        @param mask_test: testing masks.
        @param mask_train: training masks.

        """
        tf.disable_v2_behavior()

        self.input_dim = self.inp.shape[1]
        self.output_dim = self.labels.shape[1]
        self.state_dim = embed_dim
        self.tensorboard = tensorboard
        self.max_iter = max_it
        self.net = Net(self.input_dim, self.state_dim, self.output_dim)
        self.optimizer = optimizer(learning_rate, name="optim")
        self.state_threshold = threshold
        self.graph_based = graph_based
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

        self.__build()

        self.session = tf.Session(config=config)
        # self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.init_l = tf.local_variables_initializer()

        # parameter to monitor the learning via tensorboard and to save the __base
        if self.tensorboard:
            self.merged_all = tf.summary.merge_all(key='always')
            self.merged_train = tf.summary.merge_all(key='train')
            self.merged_val = tf.summary.merge_all(key='val')
            self.writer = tf.summary.FileWriter('tmp/' + param, self.session.graph)
        # self.saver = tf.train.Saver()
        # self.save_path = "tmp/" + param + "saves/__base.ckpt"
        tf.enable_v2_behavior()

    @Model._fit_in_init_model_fit
    def fit(self, num_epoch=100):
        """
        GCN - fit (step IV) \n
        Trains the GNN model.

        @param num_epoch: Number of training epochs.
        """
        tf.disable_v2_behavior()
        # train the __base
        count = 0

        ######

        for j in range(0, num_epoch):
            _, it = self.__train(inputs=self.inp, ArcNode=self.arcnode, target=self.labels, step=count,
                                 mask=self.mask_train)

            if count % 10 == 0:
                print("Epoch ", count)
                print("Training: ", self.__validate(self.inp, self.arcnode, self.labels, count, mask=self.mask_train))
                print("Test: ", self.__validate(self.inp, self.arcnode, self.labels, count, mask=self.mask_test))

                # end = time.time()
                # print("Epoch {} at time {}".format(j, end-start))
                # start = time.time()

            count = count + 1

        tf.enable_v2_behavior()

    @Model._embed_in_init_model_fit
    def embed(self):
        tf.disable_v2_behavior()
        result = self.__predict(self.inp, self.arcnode, self.labels)[2]
        tf.enable_v2_behavior()
        return result

    @staticmethod
    def fast_embed(graph: Graph, idx_labels, embed_dim=4,
                   max_it=50, optimizer=tf.train.AdamOptimizer, learning_rate=0.01, threshold=0.01, graph_based=False,
                   param=str(time.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), config=None, tensorboard=False,
                   mask_flag=True, mask_train=None, mask_test=None, num_epoch=100):
        gnn = GNN(graph)
        gnn.initialize(idx_labels)
        gnn.initialize_model(embed_dim, max_it, optimizer, learning_rate, threshold, graph_based, param, config,
                             tensorboard, mask_flag, mask_train, mask_test)
        gnn.fit(num_epoch)
        return gnn.embed()

    def __weight_variable(self, shape, nm):
        # function to initialize weights
        initial = tf.truncated_normal(shape, stddev=0.1)
        tf.summary.histogram(nm, initial, collections=['always'])
        return tf.Variable(initial, name=nm)

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
        arcnode = SparseMatrix(indices=np.stack((E[:, 0], np.arange(len(E))), axis=1),
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
        graphnode = SparseMatrix(indices=np.stack((g_ids, np.arange(len(g_ids))), axis=1),
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

    def __variableState(self):
        '''Define placeholders for input, output, state, state_old, arch-node conversion matrix'''
        # placeholder for input and output

        self.comp_inp = tf.placeholder(tf.float32, shape=(None, self.input_dim), name="input")
        self.y = tf.placeholder(tf.float32, shape=(None, self.output_dim), name="target")

        if self.mask_flag:
            self.mask = tf.placeholder(tf.float32, name="mask")

        # state(t) & state(t-1)
        self.state = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="state")
        self.state_old = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="old_state")

        # arch-node conversion matrix
        self.ArcNode = tf.sparse_placeholder(tf.float32, name="ArcNode")

        # node-graph conversion matrix
        if self.graph_based:
            self.NodeGraph = tf.sparse_placeholder(tf.float32, name="NodeGraph")
        else:
            self.NodeGraph = tf.placeholder(tf.float32, name="NodeGraph")

    def __build(self):
        '''build the architecture, setting variable, loss, training'''
        # network
        self.__variableState()
        self.loss_op = self.__loop()

        # loss
        with tf.variable_scope('loss'):
            if self.mask_flag:
                self.loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
                self.val_loss = self.net.Loss(self.loss_op[0], self.y, mask=self.mask)
            else:
                self.loss = self.net.Loss(self.loss_op[0], self.y)
                # val loss
                self.val_loss = self.net.Loss(self.loss_op[0], self.y)

            if self.tensorboard:
                self.summ_loss = tf.summary.scalar('loss', self.loss, collections=['train'])
                self.summ_val_loss = tf.summary.scalar('val_loss', self.val_loss, collections=['val'])

        # optimizer
        with tf.variable_scope('train'):
            self.grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.grads, name='train_op')
            if self.tensorboard:
                for index, grad in enumerate(self.grads):
                    tf.summary.histogram("{}-grad".format(self.grads[index][1].name), self.grads[index],
                                         collections=['always'])

        # metrics
        with tf.variable_scope('metrics'):
            if self.mask_flag:
                self.metrics = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.metrics = self.net.Metric(self.y, self.loss_op[0])

        # val metric
        with tf.variable_scope('val_metric'):
            if self.mask_flag:
                self.val_met = self.net.Metric(self.y, self.loss_op[0], mask=self.mask)
            else:
                self.val_met = self.net.Metric(self.y, self.loss_op[0])
            if self.tensorboard:
                self.summ_val_met = tf.summary.scalar('val_metric', self.val_met, collections=['always'])

    def __convergence(self, a, state, old_state, k):
        with tf.variable_scope('Convergence'):
            # body of the while cicle used to iteratively calculate state

            # assign current state to old state
            old_state = state

            # grub states of neighboring node
            gat = tf.gather(old_state, tf.cast(a[:, 1], tf.int32))

            # slice to consider only label of the node and that of it's neighbor
            # sl = tf.slice(a, [0, 1], [tf.shape(a)[0], tf.shape(a)[1] - 1])
            # equivalent code
            sl = a[:, 2:]

            # concat with retrieved state
            inp = tf.concat([sl, gat], axis=1)

            # evaluate next state and multiply by the arch-node conversion matrix to obtain per-node states
            layer1 = self.net.netSt(inp)
            state = tf.sparse_tensor_dense_matmul(self.ArcNode, layer1)

            # update the iteration counter
            k = k + 1
        return a, state, old_state, k

    def __condition(self, a, state, old_state, k):
        # evaluate condition on the convergence of the state
        with tf.variable_scope('condition'):
            # evaluate distance by state(t) and state(t-1)
            outDistance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(state, old_state)), 1) + 0.00000000001)
            # vector showing item converged or not (given a certain threshold)
            checkDistanceVec = tf.greater(outDistance, self.state_threshold)

            c1 = tf.reduce_any(checkDistanceVec)
            c2 = tf.less(k, self.max_iter)

        return tf.logical_and(c1, c2)

    def __loop(self):
        # call to loop for the state computation and compute the output
        # compute state
        with tf.variable_scope('Loop'):
            k = tf.constant(0)
            res, st, old_st, num = tf.while_loop(self.__condition, self.__convergence,
                                                 [self.comp_inp, self.state, self.state_old, k])
            if self.tensorboard:
                self.summ_iter = tf.summary.scalar('iteration', num, collections=['always'])

            if self.graph_based:
                # stf = tf.transpose(tf.matmul(tf.transpose(st), self.NodeGraph))

                stf = tf.sparse_tensor_dense_matmul(self.NodeGraph, st)
            else:
                stf = st
            out = self.net.netOut(stf)

        return out, num, stf

    def __train(self, inputs, ArcNode, target, step, nodegraph=0.0, mask=None):
        ''' train methods: has to receive the inputs, arch-node matrix conversion, target,
        and optionally nodegraph indicator '''

        # Creating a SparseTEnsor with the feeded ArcNode Matrix
        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                             dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs,
                  self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_, self.y: target, self.mask: mask}
        else:

            fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs,
                  self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
                  self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
                  self.ArcNode: arcnode_, self.y: target}
        if self.tensorboard:
            _, loss, loop, merge_all, merge_tr = self.session.run(
                [self.train_op, self.loss, self.loss_op, self.merged_all, self.merged_train],
                feed_dict=fd)
            if step % 100 == 0:
                self.writer.add_summary(merge_all, step)
                self.writer.add_summary(merge_tr, step)
        else:
            _, loss, loop = self.session.run(
                [self.train_op, self.loss, self.loss_op],
                feed_dict=fd)

        return loss, loop[1]

    def __validate(self, inptVal, arcnodeVal, targetVal, step, nodegraph=0.0, mask=None):
        """ Takes care of the validation of the __base - it outputs, regarding the set given as input,
         the loss value, the accuracy (custom defined in the Net file), the number of iteration
         in the convergence procedure """

        arcnode_ = tf.SparseTensorValue(indices=arcnodeVal.indices, values=arcnodeVal.values,
                                        dense_shape=arcnodeVal.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                             dense_shape=nodegraph.dense_shape)

        if self.mask_flag:
            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal,
                      self.mask: mask}
        else:

            fd_val = {self.NodeGraph: nodegraph, self.comp_inp: inptVal,
                      self.state: np.zeros((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.state_old: np.ones((arcnodeVal.dense_shape[0], self.state_dim)),
                      self.ArcNode: arcnode_,
                      self.y: targetVal}

        if self.tensorboard:
            loss_val, loop, merge_all, merge_val, metr = self.session.run(
                [self.val_loss, self.loss_op, self.merged_all, self.merged_val, self.metrics], feed_dict=fd_val)
            self.writer.add_summary(merge_all, step)
            self.writer.add_summary(merge_val, step)
        else:
            loss_val, loop, metr = self.session.run(
                [self.val_loss, self.loss_op, self.metrics], feed_dict=fd_val)
        return loss_val, metr, loop[1]

    def __evaluate(self, inputs, st, st_old, ArcNode, target):
        '''evaluate method with initialized state -- not used for the moment: has to receive the inputs,
        initialization for state(t) and state(t-1),
        arch-node matrix conversion, target -- gives as output the accuracy on the set given as input'''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)

        fd = {self.comp_inp: inputs, self.state: st, self.state_old: st_old,
              self.ArcNode: arcnode_, self.y: target}
        _ = self.session.run([self.init_l])
        met = self.session.run([self.metrics], feed_dict=fd)
        return met

    def __evaluate(self, inputs, ArcNode, target, nodegraph=0.0):
        '''evaluate methods: has to receive the inputs,  arch-node matrix conversion, target
         -- gives as output the accuracy on the set given as input'''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        if self.graph_based:
            nodegraph = tf.SparseTensorValue(indices=nodegraph.indices, values=nodegraph.values,
                                             dense_shape=nodegraph.dense_shape)

        fd = {self.NodeGraph: nodegraph, self.comp_inp: inputs,
              self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_, self.y: target}
        _ = self.session.run([self.init_l])
        met = self.session.run([self.metrics], feed_dict=fd)
        return met

    def __predict(self, inputs, st, st_old, ArcNode):
        ''' predict methods with initialized state -- not used for the moment:: has to receive the inputs,
         initialization for state(t) and state(t-1),
         arch-node matrix conversion -- gives as output the output values of the output function (all the nodes output
         for all the graphs (if node-based) or a single output for each graph (if graph based) '''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        fd = {self.comp_inp: inputs, self.state: st, self.state_old: st_old,
              self.ArcNode: arcnode_}
        pr = self.session.run([self.loss_op], feed_dict=fd)
        return pr[0]

    def __predict(self, inputs, ArcNode, nodegraph=0.0):
        ''' predict methods: has to receive the inputs, arch-node matrix conversion -- gives as output the output
         values of the output function (all the nodes output
         for all the graphs (if node-based) or a single output for each graph (if graph based) '''

        arcnode_ = tf.SparseTensorValue(indices=ArcNode.indices, values=ArcNode.values,
                                        dense_shape=ArcNode.dense_shape)
        fd = {self.comp_inp: inputs, self.state: np.zeros((ArcNode.dense_shape[0], self.state_dim)),
              self.state_old: np.ones((ArcNode.dense_shape[0], self.state_dim)),
              self.ArcNode: arcnode_}
        pr = self.session.run([self.loss_op], feed_dict=fd)
        return pr[0]


class Net:
    # class to define state and output network

    def __init__(self, input_dim, state_dim, output_dim):
        # initialize weight and parameter

        self.EPSILON = 0.00000001

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.state_input = self.input_dim - 1 + state_dim  # removing the id_ dimension

        #### TO BE SET ON A SPECIFIC PROBLEM
        self.state_l1 = 5
        self.state_l2 = self.state_dim

        self.output_l1 = 5
        self.output_l2 = self.output_dim

    def netSt(self, inp):
        with tf.variable_scope('State_net'):
            layer1 = tf.layers.dense(inp, self.state_l1, activation=tf.nn.tanh)
            layer2 = tf.layers.dense(layer1, self.state_l2, activation=tf.nn.tanh)

            return layer2

    def netOut(self, inp):
        layer1 = tf.layers.dense(inp, self.output_l1, activation=tf.nn.tanh)
        layer2 = tf.layers.dense(layer1, self.output_l2, activation=tf.nn.softmax)

        return layer2

    def Loss(self, output, target, output_weight=None, mask=None):
        # method to define the loss function
        # lo = tf.losses.softmax_cross_entropy(target, output)
        output = tf.maximum(output, self.EPSILON, name="Avoiding_explosions")  # to avoid explosions
        xent = -tf.reduce_sum(target * tf.log(output), 1)

        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        xent *= mask
        lo = tf.reduce_mean(xent)
        return lo

    def Metric(self, target, output, output_weight=None, mask=None):
        # method to define the evaluation metric

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask

        return tf.reduce_mean(accuracy_all)
