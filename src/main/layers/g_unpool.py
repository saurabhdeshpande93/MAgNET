import tensorflow as tf
from tensorflow import keras


class G_Unpool(keras.layers.Layer):

    """

    Un-pooling layer

    Input    : Tensorflow layer with shape (None, units)        ;  units        = n_channels*nodes
    Output   : Tensorflow layer with shape (None, unpool_units) ;  unpool_units = n_channels*unpool_nodes

    :arguments

    subgraph = This is a list of tensors(tf.int32) containing the subgraph over which pooling has to be performed
    nodes    = number of nodes in the input graph

    """

    def __init__(self, subgraph, nodes = None):
        super(G_Unpool, self).__init__()
        self.subgraph = subgraph
        self.nodes = nodes

    def call(self, inputs):

        batch_dim = tf.shape(inputs)[0]
        units = inputs.shape[-1]

        n_cliques = len(self.subgraph)
        n_channels = int(units/n_cliques)

        pooled_out = tf.reshape(inputs, [batch_dim, n_channels, n_cliques])

        unpool_out = tf.zeros([batch_dim, n_channels, self.nodes])

        # i-th coloumn of pooled out should replicate to columns corresponding to nodes of that element in the unpooled out
        for i in range(n_cliques):
            node_list = self.subgraph[i]
            pool_length = tf.shape(node_list)[0]
            indices = tf.stack([tf.tile(tf.repeat(tf.range(batch_dim), pool_length), [n_channels]),
                                tf.repeat(tf.range(n_channels), batch_dim * pool_length)], axis=1)
            indices = tf.concat([indices,tf.expand_dims(tf.tile(node_list, [batch_dim*n_channels]),axis=-1)], axis=1)

            indices = tf.reshape(indices, [batch_dim, n_channels, pool_length, 3])

            updates = tf.gather(pooled_out, [i], axis=2)
            updates = tf.repeat(updates, pool_length, axis=2)

            unpool_out = tf.tensor_scatter_nd_add(unpool_out, indices, updates)


        unpool_out = tf.reshape(unpool_out,[batch_dim,self.nodes*n_channels])

        return unpool_out


    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'subgraph': self.subgraph,
            'nodes': self.nodes,
        })
        return config
