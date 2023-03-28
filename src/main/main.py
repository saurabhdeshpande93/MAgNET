import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layers.pool_utils import pooled_adjacencies
from models import Models
from utils import *
from global_vars import *

## case = '2dlshape', '2dhole', '3dbeam', '3dbreast'
case = '2dhole'

## network type = 'magnet', 'cnn'
type = 'magnet'

## load train/test datasets
data = get_data(case, type)

dof, dim = dofdim[type][case]

if type == 'magnet':
    A = get_adjacency(case)
    no_poolings, poolseed = magnet_pool_details[case]
    pooled_adjs, subgraphs = pooled_adjacencies(A, no_poolings=no_poolings, poolseed=poolseed)

    channels = magnet_channels[case]
    input_shape = (dof,)

    network = Models(input_shape, channels)

    if case == '2dhole' or case == '2dlshape':
        model = network.magnet_2d(pooled_adjs, subgraphs)
    elif case == '3dbeam':
        model = network.magnet_3dbeam(pooled_adjs, subgraphs)
    else:
        model = network.magnet_3dbreast(pooled_adjs, subgraphs)


else:
    input_shape = cnn_input_shapes[case]
    channels = cnn_channels[case]
    network = Models(input_shape, channels)

    model = network.cnn_2d() if case == '2dlshape' else network.cnn_3d()


training = False

## load the pretrained weights
if training == False:
    model.load_weights(weights_path+case+'_'+type+'.h5')
    print("=== Pretrained weights are assigned to the model ===")

## for training = True
else:
    no_epochs = epochs[type][case]
    new_weights_path = weights_path+case+'_'+type+'new.h5'
    lr_scheduler = lrate_scheduler(case)
    network.train(model, data, new_weights_path, lr_scheduler, no_epochs)
    model.load_weights(new_weights_path)

## Compute predictions
test_features = data[-2]
n_test = len(test_features)
predictions = model.predict(test_features, batch_size=4)
if type == 'cnn': predictions = predictions.reshape((n_test,dof))

## Convert degrees of freedom ordering to Acegen format
predictions = reorder_dof(predictions,dof,dim)
X_test = reorder_dof(data[-2].reshape((n_test,dof)),dof,dim)
Y_test = reorder_dof(data[-1].reshape((n_test,dof)),dof,dim)
## Remove zero paded region for the 2D L-shape CNN case
if case == '2dlshape' and type == 'cnn':
    predictions = remove_pad(predictions)
    X_test = remove_pad(X_test)
    Y_test = remove_pad(Y_test)


np.save(prediction_path+case+'_'+type+'_predicts.npy', predictions)
np.save(prediction_path+case+'_'+type+'_x.npy', X_test)
np.save(prediction_path+case+'_'+type+'_y.npy', Y_test)

print("=== Predictions are saved in : {}".format(prediction_path))

## Set to True to print the model
network.print_summary(model, print=False)
