import tensorflow as tf
import numpy as np


def adjacency_pool(adjacency, pool_type: str, seed=10):

    """

    Function for creating pooled adjacency matrix along with list of subgraphs over which pooling is to be performed.

    :argument      adjacency: A numpy array of adjacency matrix
    :argument      pool_type: String defining type of pooling
                          'clique'  -  Pooling over non-overlapping cliques of a graph
                          'element' -  Pooling over elements of a FEM mesh
    :argument      seed     : Integer for numpy random seed.

    :return:       A_pool   : Pooled adjacency matrix
    :return:       subgraphs: List of tensors representing nodes over which pooling is to be performed

    """

    if pool_type == 'clique':

        A_reduced = np.copy(adjacency)
        subgraphs = []
        node_list = np.arange(len(A_reduced))
        np.random.seed(seed)

        while ~(node_list == 0.0).all():

            rand_node = np.random.choice(node_list)

            graph = np.array([rand_node])

            connected_nodes = np.squeeze(np.array(np.nonzero(A_reduced[rand_node])))
            connected_nodes = np.delete(connected_nodes, np.where(connected_nodes == rand_node))

            # connections = [p for p in itertools.combinations(connected_nodes, 2)]

            for node in connected_nodes:
                if (adjacency[graph, node] == 1.0).all():
                    graph = np.append(graph, node)

            A_reduced[:, graph] = 0.
            A_reduced[graph, :] = 0.

            subgraphs.append(graph)

            node_list = np.asarray([i for i in node_list if i not in graph])

        #Corss check if you have a overlapping node in the subgraph
        for i in range(len(subgraphs)):
            for j in range(i, len(subgraphs)):
                if i != j and np.in1d(subgraphs[i], subgraphs[j]).any():
                    raise ValueError('Common node found')

        A_pool = new_adjacency(subgraphs,adjacency)

        subgraphs = [tf.convert_to_tensor(graph, dtype=tf.int32) if graph.ndim != 0 else tf.expand_dims(
            tf.convert_to_tensor(graph, dtype=tf.int32), axis=0) for graph in subgraphs]

    # We pool over each element of the finite element mesh, not implemented in the paper
    elif pool_type == 'element':

        no_ele = len(e_connect)

        A_pool = np.zeros((no_ele, no_ele))

        for i in range(no_ele):
            for j in range(no_ele):
                if bool(set(e_connect[i]) & set(e_connect[j])):
                    A_pool[i, j] = 1

        subgraphs = tf.convert_to_tensor(e_connect, dtype=tf.int32)


    else:
        raise ValueError('This type of pooling is not implemented')


    return A_pool, subgraphs



def new_adjacency(subgraphs, adjacency):

    '''

    Creates a pooled adjacency using subghrap connections and parent adjacency matrix

    '''

    new_size = len(subgraphs)
    A_pool = np.zeros((new_size, new_size))

    for i in range(new_size):
        for j in range(new_size):
            slice1 = adjacency[subgraphs[i], :]

            if slice1.ndim == 1:
                slice1 = np.expand_dims(slice1, axis=0)

            slice2 = slice1[:, subgraphs[j]]

            if ~(slice2 == 0.0).all():
                A_pool[i, j] = 1

    return A_pool




def pooled_adjacencies(adjacency, no_poolings, poolseed):

    '''
    Generates multiple pooled adjacencies

    :argument      adjacency   : Adjacency matrix of the input graph (adjacency of first graph U-Net level)
    :argument      no_poolings : Number of times pooling to be performed (number of pooling operations in graph U-Net)
    :argument      poolseed    : Random seed for the first pooling operation

    :return:       pooled_adjs : Adjacencies of pooled graphs
    :return:       subgraphs   : Subgraphs (List of nodes over which pooling performed) for all pooled adjacencies

    '''

    pooled_adjs = [adjacency]
    subgraphs = []

    for i in range(no_poolings):
        if i==0:
            A_pooled, subgraph = adjacency_pool(pooled_adjs[-1],pool_type='clique', seed=poolseed)
            pooled_adjs.append(A_pooled)
            subgraphs.append(subgraph)
        else:
            A_pooled, subgraph = adjacency_pool(pooled_adjs[-1],pool_type='clique')
            pooled_adjs.append(A_pooled)
            subgraphs.append(subgraph)

    pooled_adjs = [tf.convert_to_tensor(adj,dtype=tf.float32) for adj in pooled_adjs]

    return pooled_adjs, subgraphs
