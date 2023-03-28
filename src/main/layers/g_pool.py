import tensorflow as tf
from tensorflow import keras


class G_Pool(keras.layers.Layer):

    """

    Pooling layer

    Input    : Tensorflow layer with shape  (None, units)       ;   units = n_channels*nodes
    Output   : Tensorflow layer with shape  (None, pool_units)  ;   unpool_units = n_channels*pool_nodes

    :argument

    subgraph = This is a list of tensors(tf.int32) containing the subgraphs over which pooling has to be performed
    nodes    = number of nodes for the unpooled graph (for the originally pooled graph)

    """

    def __init__(self,subgraph, nodes = None):
        super(G_Pool, self).__init__()
        self.subgraph = subgraph
        self.nodes = nodes

    def call(self, inputs):

        batch_dim = tf.shape(inputs)[0]
        units = inputs.shape[-1]

        n_channels = int(units/self.nodes)
        n_cliques = len(self.subgraph)

        batch_layer = tf.reshape(inputs, [batch_dim, n_channels, self.nodes])

        out = tf.range(n_cliques, dtype=tf.float32)
        out = tf.reshape(out, [1, n_cliques])
        out = tf.tile(out, tf.constant([n_channels, 1], tf.int32))
        out = tf.tile(tf.expand_dims(out, 0), [batch_dim, 1, 1])
        pooled_out = out

        for i in range(n_cliques):
            sliced_tensor = tf.gather(batch_layer, self.subgraph[i], axis=2)
            indices = tf.where(tf.equal(out, tf.constant(i, dtype=tf.float32)))
            indices = tf.reshape(indices, [batch_dim, n_channels, 3])
            pooled_out = tf.tensor_scatter_nd_update(pooled_out, indices, tf.reduce_max(sliced_tensor, axis=2))

        pooled_out = tf.reshape(pooled_out,[batch_dim,n_cliques*n_channels])

        return pooled_out


    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'subgraph': self.subgraph,
            'nodes': self.nodes,
        })
        return config
