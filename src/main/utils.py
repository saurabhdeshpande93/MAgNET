import numpy as np
import pandas as pd
import itertools
from global_vars import *


def get_data(case, type):

    '''
    Import the datasets (numpy arrays) used in the paper

    Shape of X_train/Y_train = [n_train, dof]
    Shape of X_test/Y_test = [n_test, dof]

    dofs are ordered as : [1x, 2x, ... , nx, 1y, 2y, ... , ny, 1z, 2z, ... , nz]

    '''

    print_magnet()

    if type == 'cnn' and  case=='2dhole' or type == 'cnn' and case =='3dbreast':
        raise ValueError('This case is not implemented')


    print("=== Loading dataset for the "+ type.upper() +" network and "+ case + " case === ")

    if case == '2dlshape' and type == 'cnn': case += '_padded'

    X_train = np.load(datapath+'features_train_'+str(case)+'.npy')
    Y_train = np.load(datapath+'labels_train_'+str(case)+'.npy')

    X_test = np.load(datapath+'features_test_'+str(case)+'.npy')
    Y_test = np.load(datapath+'labels_test_'+str(case)+'.npy')

    if type == 'cnn':
        # Need to reshape data to make it compatible with convolution layers
        n_train = len(X_train)
        n_test = len(X_test)

        if case == '3dbeam':
            dim, n_x, n_y, n_z = cnn_input_shapes[case]

            X_train = X_train.reshape(n_train, dim, n_x, n_y, n_z)
            Y_train = Y_train.reshape(n_train, dim, n_x, n_y, n_z)

            X_test = X_test.reshape(n_test, dim, n_x, n_y, n_z)
            Y_test = Y_test.reshape(n_test, dim, n_x, n_y, n_z)

        else:
            case = case.replace('_padded','')

            dim, n_x, n_y = cnn_input_shapes[case]

            X_train = X_train.reshape(n_train, dim, n_x, n_y)
            Y_train = Y_train.reshape(n_train, dim, n_x, n_y)

            X_test = X_test.reshape(n_test, dim, n_x, n_y)
            Y_test = Y_test.reshape(n_test, dim, n_x, n_y)

    return [X_train, Y_train, X_test, Y_test]



def get_adjacency(case: str):

    '''
    Generates adjacancy matrix (inclues cross edges in elements) using the connectivity matrix.

    '''

    # dataframe for elemenet connectivity file
    df_adj = pd.read_csv('connectivity/connect_'+str(case)+'.csv', header=None)

    element_connectivity = df_adj.to_numpy() - 1  #Acegen numbering starts from 1
    n_elements = element_connectivity.shape[0]
    n_nodes = np.max(element_connectivity) + 1

    A = np.zeros((n_nodes,n_nodes))

    for i in range(n_elements):

        element = element_connectivity[i]

        t = [p for p in itertools.product(element, repeat=2)]

        for j in range(len(t)):
            A[t[j]] = 1

    return A


def nth_adjacency(A, n):

    '''
    Gives n-th hop for the given adjacency matrix

    '''

    A_n = A

    if n == 1:
        A_n = A

    else:

        for i in range(n - 1):
            A_n = np.dot(A_n, A)

        non_zero = np.count_nonzero(A_n)

        rows, cols = np.nonzero(A_n)
        A_n[rows, cols] = np.ones(non_zero)

    return A_n


def lrate_scheduler(case):

    '''
    Learning rate schedulers used in the paper. 2D and 3D cases use different schedulers

    '''
    if case == '2dlshape' or case =='2dhole':
        def scheduler(epoch, lr):

            if epoch < 10:
                lr = lr
            elif 10 < epoch < 200:
                lr = lr  - (0.0001 - 0.00001)/(190)
            elif 200 < epoch < 900:
                lr = lr- (0.00001 - 0.000001)/(700)
            else:
                lr = lr
            return lr

    else:
        def scheduler(epoch, lr):

            if epoch < 10:
                lr = lr
            elif 10 <= epoch < 100:
                lr = lr  - (0.0001 - 0.000001)/(90)
            else:
                lr = lr
            return lr

    return scheduler


def reorder_dof(predictions, dof, dim):

    '''
    Changes dof oredering to the Acegen format for entire predictions.
    Refer to original_order function below.

    Can be efficiently done as:
    predictions.reshape(len(predictions), dim, int(dof/dim)).transpose(0,2,1).reshape(len(predictions),dof)

    '''

    reorder_predictions = np.copy(predictions)

    for i in range(len(predictions)):
        reorder_predictions[i] = orginal_order(predictions[i],dof,dim)

    return reorder_predictions


def orginal_order(array, dof, dim):

    '''
    Inputs:

    - Dof array in the form (1x, 2x, ... , nx, 1y, 2y ..., ny, 1z, ... ,nz)

    Outputs:

    - Array with original Acegen ordering  (1x, 1y, 1z, 2x, 2y, 2z ...,nx,ny,nz)

    Can be efficiently done as: array.reshape(dim,int(dof/dim)).transpose().flatten()

    '''

    original_dof = np.zeros((dof,))

    if dim==2:

      dof_x = array[0:int(dof/dim)]
      dof_y = array[int(dof/dim):dof]

      for i in range(int(dof/dim)):
         original_dof[2*i] = dof_x[i]
         original_dof[2*i+1] = dof_y[i]


    elif dim==3:

      dof_x = array[0:int(dof/dim)]
      dof_y = array[int(dof/dim):int(2*dof/dim)]
      dof_z = array[int(2*dof/dim):dof]

      for i in range(int(dof/dim)):
         original_dof[3*i]   = dof_x[i]
         original_dof[3*i+1] = dof_y[i]
         original_dof[3*i+2] = dof_z[i]

    return original_dof

def remove_pad(predictions):

    '''
    To remove dofs in the 0 paded region for the L-shape cnn case.

    unpad_map_2dlshape.npy is the arrays for indices of non-zero padded region of the padded
    2D L-shape domain.

    '''

    unpad_indices = np.load(srcpath+'/main/connectivity/unpad_map_2dlshape.npy')

    return predictions[:,unpad_indices]


def max_disp_index(predictions, dof, dim):

    '''
    Gives array of maximum nodal displacements of all test examples.
    '''

    n_test = len(predictions)
    max_disps = np.zeros(n_test)

    for i in range(n_test):
        pred =  predictions[i]
        pred =  pred.reshape(int(dof/dim),dim)
        pred_norm = np.linalg.norm(pred,axis=1)
        max_disps[i] = np.max(abs(pred_norm))

    return np.argmax(max_disps)


def print_magnet():
    print("\n")
    print("         ___  ___  ___        _   _  _____ _____ ")
    print("         |  \/  | / _ \      | \ | ||  ___|_   _|")
    print("         | .  . |/ /_\ \ __ _|  \| || |__   | |  ")
    print("         | |\/| ||  _  |/ _` | . ` ||  __|  | |  ")
    print("         | |  | || | | | (_| | |\  || |___  | |  ")
    print("         \_|  |_/\_| |_/\__, \_| \_/\____/  \_/  ")
    print("                         __/ |                   ")
    print("                        |___/                    ")
    print("\n")
