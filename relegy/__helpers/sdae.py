"""
@author madhumita
@author lukaszbrzozowski
The source code: https://github.com/MadhumitaSushil/SDAE
It was updated and slightly modified to allow for fine-tuning of DNGR
"""

import os

import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential

from . import nn_utils

os.environ['THEANO_FLAGS'] = "device=gpu1,floatX=float32"
os.environ['KERAS_BACKEND'] = "theano"
os.environ['PYTHONHASHSEED'] = '0'


class SDAE(object):
    '''
    Implements stacked denoising autoencoders in Keras, without tied weights.
    To read up about the stacked denoising autoencoder, check the following paper:

    Vincent, Pascal, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol.
    "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."
    Journal of Machine Learning Research 11, no. Dec (2010): 3371-3408.
    '''

    def __init__(self, n_layers=1, n_hid=500, dropout=0.05, enc_act='sigmoid', dec_act='linear', bias=True,
                 loss_fn='mse', batch_size=32, nb_epoch=300, optimizer='adam', verbose=1, random_state=None):
        '''
        Initializes parameters for stacked denoising autoencoders
        @param n_layers: number of layers, i.e., number of autoencoders to stack on top of each other.
        @param n_hid: list with the number of hidden nodes per layer. If only one value specified, same value is used for all the layers
        @param dropout: list with the proportion of data_in nodes to mask at each layer. If only one value is provided, all the layers share the value.
        @param enc_act: list with activation function for encoders at each layer. Typically sigmoid.
        @param dec_act: list with activation function for decoders at each layer. Typically the same as encoder for binary data_in, linear for real data_in.
        @param bias: True to use bias value.
        @param loss_fn: The loss function. Typically 'mse' is used for real values. Options can be found here: https://keras.io/objectives/
        @param batch_size: mini batch size for gradient update
        @param nb_epoch: number of epochs to train each layer for
        @param optimizer: The optimizer to use. Options can be found here: https://keras.io/optimizers/
        @param verbose: 1 to be verbose
        '''
        n_layers: int
        n_hid: list
        enc_act: list
        dec_act: list
        bias: bool
        dropout: list
        loss_fn: str
        batch_size: int
        nb_epoch: int
        optimizer: int
        verbose: bool
        self.n_layers = n_layers

        # if only one value specified for n_hid, dropout, enc_act or dec_act, use the same parameters for all layers.
        self.n_hid, self.dropout, self.enc_act, self.dec_act = self._assert_input(n_layers, n_hid, dropout, enc_act,
                                                                                  dec_act)
        self.bias = bias

        self.loss_fn = loss_fn

        self.batch_size = batch_size

        self.nb_epoch = nb_epoch

        self.optimizer = optimizer

        self.verbose = verbose

        self.random_state = random_state

    def get_pretrained_sda(self, data_in, get_enc_model=True, get_enc_dec_model=False, model_layers=None):
        '''
        Pretrains layers of a stacked denoising autoencoder to generate low-dimensional representation of data.
        Returns a Sequential __base with the Dropout layer and pretrained encoding layers added sequentially.
        Optionally, we can return a list of pretrained sdae models by setting get_enc_model to False.
        Additionally, returns dense representation of input, validation and test data.
        This dense representation is the value of the hidden node of the last layer.
        The cur_model be used in supervised task by adding a classification/regression layer on top,
        or the dense pretrained data can be used as input of another cur_model.
        @param data_in: input data (scipy sparse matrix supported)
        @param get_enc_model: True to get a Sequential __base with Dropout and encoding layers from SDAE.
                              If False, returns a list of all the encoding-decoding models within our stacked denoising autoencoder.
        @param get_enc_dec_model: If true, returns the __base built with the decoder layers. Overrides get_enc_model
        @param model_layers: Pretrained cur_model layers, to continue training pretrained model_layers, if required
        '''
        if model_layers is not None:
            self.n_layers = len(model_layers)
        else:
            model_layers = [None] * self.n_layers

        encoders = []
        decoders = []

        recon_mse = 0

        for cur_layer in range(self.n_layers):

            if model_layers[cur_layer] is None:
                input_layer = Input(shape=(data_in.shape[1],))

                # masking input data to learn to generalize, and prevent identity learning
                dropout_layer = Dropout(self.dropout[cur_layer])
                in_dropout = dropout_layer(input_layer)

                encoder_layer = Dense(units=self.n_hid[cur_layer], kernel_initializer='glorot_uniform',
                                      activation=self.enc_act[cur_layer], name='encoder' + str(cur_layer),
                                      use_bias=self.bias)
                encoder = encoder_layer(in_dropout)

                n_out = data_in.shape[1]  # same no. of output units as input units (to reconstruct the signal)

                decoder_layer = Dense(units=n_out, use_bias=self.bias, kernel_initializer='glorot_uniform',
                                      activation=self.dec_act[cur_layer], name='decoder' + str(cur_layer))
                decoder = decoder_layer(encoder)

                cur_model = Model(input_layer, decoder)

                cur_model.compile(loss=self.loss_fn, optimizer=self.optimizer)

            else:
                cur_model = model_layers[cur_layer]

            print("Training layer " + str(cur_layer))

            hist = cur_model.fit(x=nn_utils.batch_generator(
                data_in, data_in,
                batch_size=self.batch_size,
                shuffle=True,
                seed = self.random_state
            ),
                epochs=self.nb_epoch,
                steps_per_epoch=data_in.shape[0],
                verbose=self.verbose
            )

            print("Layer " + str(cur_layer) + " has been trained")

            model_layers[cur_layer] = cur_model
            encoder_layer = cur_model.layers[-2]
            decoder_layer = cur_model.layers[-1]

            encoders.append(encoder_layer)
            decoders.append(decoder_layer)

            if cur_layer == 0:
                recon_mse = self._get_recon_error(cur_model, data_in, n_out=cur_model.layers[-1].output_shape[1])

            data_in = self._get_intermediate_output(cur_model, data_in, n_layer=2, train=0, n_out=self.n_hid[cur_layer],
                                                    batch_size=self.batch_size)
            assert data_in.shape[1] == self.n_hid[cur_layer], "Output of hidden layer not retrieved"

        if get_enc_model or get_enc_dec_model:
            final_model = self._build_model_from_encoders(encoders, dropout_all=False)  # , final_act_fn= final_act_fn)
            if get_enc_dec_model:
                final_model = self._build_model_from_encoders_and_decoders(final_model, decoders)
        else:
            final_model = model_layers

        return final_model, data_in, recon_mse

    def _build_model_from_encoders(self, encoding_layers, dropout_all=False) -> Sequential:
        '''
        Builds a deep NN __base that generates low-dimensional representation of input, based on pretrained layers.
        @param encoding_layers: pretrained encoder layers
        @param dropout_all: True to include dropout layer between all layers. By default, dropout is only present for input.
        @return __base with each encoding layer as a layer of a NN
        '''
        model = Sequential()
        model.add(Dropout(self.dropout[0], input_shape=(encoding_layers[0].input_shape[1],)))

        for i in range(len(encoding_layers)):
            if i and dropout_all:
                model.add(Dropout(self.dropout[i]))

            model.add(encoding_layers[i])

        return model

    @staticmethod
    def _build_model_from_encoders_and_decoders(enc_model, decoding_layers) -> Sequential:
        """
        Build __base with both the encoded and decoded groups of layers
        :param enc_model: Model created with _build_model_from_encoders
        :param decoding_layers: pretrained decoder layers
        :return: __base with the encoding and decoding layers as NN
        """
        model = enc_model
        for i in range(1, len(decoding_layers) + 1):
            model.add(decoding_layers[-i])

        return model

    @staticmethod
    def _assert_input(n_layers, n_hid, dropout, enc_act, dec_act):
        """
        If the hidden nodes, dropout proportion, encoder activation function or decoder activation function is given, it uses the same parameter for all the layers.
        Errors out if there is a size mismatch between number of layers and parameters for each layer.
        """

        if type(n_hid) == int:
            n_hid = [n_hid] * n_layers
        else:
            assert (type(n_hid[0]) == int or type(n_hid[0]) == np.int32)

        if type(dropout) == int or type(dropout) == float:
            dropout = [dropout] * n_layers

        if type(enc_act) == str:
            enc_act = [enc_act] * n_layers

        if type(dec_act) == str:
            dec_act = [dec_act] * n_layers

        assert (n_layers == len(n_hid) == len(dropout) == len(enc_act) == len(
            dec_act)), "Please specify as many hidden nodes, dropout proportion on input, and encoder and decoder activation function, as many layers are there, using list data structure"

        return n_hid, dropout, enc_act, dec_act

    def  _get_intermediate_output(self, model, data_in, n_layer, train, n_out, batch_size, dtype=np.float32):
        '''
        Returns output of a given intermediate layer in a __base
        @param model: __base to get output from
        @param data_in: sparse representation of input data
        @param n_layer: the layer number for which output is required
        @param train: (0/1) 1 to use training config, like dropout noise.
        @param n_out: number of output nodes in the given layer (pre-specify so as to use generator function with sparse matrix to get layer output)
        @param batch_size: the num of instances to convert to dense at a time
        @return value of intermediate layer
        '''
        data_out = np.zeros(shape=(data_in.shape[0], n_out))

        x_batch_gen = nn_utils.x_generator(data_in, batch_size=batch_size, shuffle=False, seed=self.random_state)
        stop_iter = int(np.ceil(data_in.shape[0] / batch_size))

        for i in range(stop_iter):
            cur_batch, cur_batch_idx = next(x_batch_gen)
            data_out[cur_batch_idx, :] = self._get_nth_layer_output(model, n_layer, X=cur_batch, train=train)

        return data_out.astype(dtype, copy=False)

    def _get_nth_layer_output(self, model, n_layer, X, train=1):
        '''
        Returns output of nth layer in a given __base.
        @param model: keras __base to get an intermediate value out of
        @param n_layer: the layer number to get the value of
        @param X: input data for which layer value should be computed and returned.
        @param train: (1/0): 1 to use the same setting as training (for example, with Dropout, etc.), 0 to use the same setting as testing phase for the __base.
        @return the value of n_layer in the given __base, input, and setting
        '''
        partial_model = Model(model.inputs, model.layers[n_layer].output)
        return partial_model([X], training=train)

    def _get_recon_error(self, model, data_in, n_out):
        """
        Return reconstruction squared error at individual nodes, averaged across all instances.
        @param model: trained __base
        @param data_in: input data to reconstruct
        @param n_out: number of __base output nodes
        """
        train_recon = self._get_intermediate_output(model, data_in, n_layer=-1, train=0, n_out=n_out,
                                                    batch_size=self.batch_size)
        recon_mse = np.mean(np.square(train_recon - data_in), axis=0)

        recon_mse = np.ravel(recon_mse)

        return recon_mse
