import numpy as np
import sys
sys.path.append('../main/')
from utils import get_data, reorder_dof, remove_pad, max_disp_index
from global_vars import *

## case = '2dlshape', '2dhole', '3dbeam', '3dbreast'
case = '2dhole'
## network type = 'magnet', 'cnn'
type = 'magnet'

## Import predictions
predictions = np.load(prediction_path+case+'_'+type+'_predicts.npy')

## Import deatures/labels and reorder dofs as per the Acegen compatible format
dof, dim = dofdim[type][case]

data = get_data(case, type)
test_features, test_labels = data[-2], data[-1]

test_labels = reorder_dof(test_labels.reshape((len(test_labels), dof)), dof, dim)
test_features = reorder_dof(test_features.reshape((len(test_features), dof)), dof, dim)

## Remove zero padded region for L-shape CNN case
if case == '2dlshape' and type == 'cnn':
    test_labels   = remove_pad(test_labels)
    test_features = remove_pad(test_features)
    dof = 160 #unpaded dofs

## Absolute error of prediction
error = np.abs(predictions - test_labels)

print("=== Mean error of the test set is ", np.mean(error))
print("=== Max error of the test set is ", np.max(error))

## Find out the example with maximum nodal displacement
index = max_disp_index(predictions, dof, dim)

## Save prediction of single example to visualise it in Acegen
test_viz = np.vstack((test_features[index],predictions[index],test_labels[index],error[index])).transpose()
np.savetxt('visualisation/examples/'+str(case)+str(type)+'.csv', test_viz, delimiter=",")

print("=== Example to be visualised is saved in : {}".format(visualisation_path))
