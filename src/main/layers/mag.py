import tensorflow as tf
from tensorflow import keras
from keras import activations
import numpy as np

class MAg(keras.layers.Layer):

    """

    Multichannel Aggregation Layer

    - Input   : Tensorflow layer of shape (None, n*channels[i-1])
    - Output  : Tensorflow layer of shape (None, n*channels)

    where,
    'channels[i-1]' : number of channels in the input layer
    'None'          : stands for the batch size.
    'n'             : number of nodes in the graph

    :arguments

    channels     =  number of channels in the input layer
    adjacency    =  adjacency matrix for the input graph

    """

    def __init__(self, channels, adjacency, kernel_initializer="glorot_uniform",
                 bias_initializer="glorot_uniform",
                 activation=None, use_bias=True):
        super(MAg, self).__init__()
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.adjacency = adjacency
        self.nonzero = np.count_nonzero(adjacency)
        self.n_nodes = adjacency.shape[-1]


    def build(self, input_shape):

        in_dim = input_shape[-1]
        in_chan = int(in_dim/self.n_nodes)

        out_dim = self.channels*self.n_nodes
        ker_dim = self.channels*self.nonzero*in_chan


        self.kernel = self.add_weight(
            shape=(ker_dim,),
            initializer=self.kernel_initializer,
            name='kernel',
            trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(out_dim,), initializer=self.bias_initializer,name='bias', trainable=True)


    def call(self, inputs):

        in_dim = inputs.shape[-1]
        out_dim = self.channels*self.n_nodes
        in_chan = int(in_dim/self.n_nodes)

        mul = tf.constant([in_chan,self.channels],tf.int32)
        ker_wt = tf.tile(self.adjacency, mul)

        where = tf.not_equal(ker_wt, tf.constant(0, dtype=tf.float32))
        indices = tf.where(where)

        ker_wt = tf.tensor_scatter_nd_update(ker_wt, indices, self.kernel)


        out = tf.matmul(inputs, ker_wt)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)

        return out


    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'channels': self.channels,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'adjacency': self.adjacency,
            'nonzero': self.nonzero,
            'n_nodes': self.n_nodes,
        })
        return config
