
'''

All global variables are defined in this file.

'''

## All paths
import os

srcpath = os.path.dirname(os.getcwd())
datapath = srcpath + "/data/FEMData/"
weights_path = srcpath + "/data/saved_models/"
prediction_path = srcpath + "/data/predictions/"
visualisation_path = srcpath + "/postprocess/visualisation/"


## [Degrees of freedom(dof) and dimensionality(dim) of the problem].
dofdim = { 'magnet': {'2dlshape':   [160, 2],
                      "2dhole"  :   [198, 2],
                      "3dbeam"  :   [12096, 3],
                      "3dbreast":   [3105, 3]},
              'cnn': {'2dlshape':   [256, 2],  #CNN 2d L-shape has padded dofs
                      "3dbeam"  :   [12096, 3]}
                      }

## No. of channels at respective graph U-NET level of MAgNET
magnet_channels = {
                      "2dlshape":   [16, 32, 64, 128],
                      "2dhole"  :   [8, 16, 32, 64],
                      "3dbeam"  :   [6, 6, 12, 24, 48, 96],
                      "3dbreast":   [6, 12, 12, 24, 48]
                  }

## [Number of poolings, random seed for the first pooling layer]
magnet_pool_details = {
                      "2dlshape":   [3, 44],
                      "2dhole"  :   [3, 632],
                      "3dbeam"  :   [5, 827],
                      "3dbreast":   [4, 96]
                     }

## Grid mesh dimensions
cnn_input_shapes = {
                      "2dlshape":   [2, 16, 8],
                      "3dbeam"  :   [3, 28, 12, 12],
                    }

## No. of channels at respective CNN U-NET level
cnn_channels = {
                      "2dlshape":   [64, 128, 512],
                      "3dbeam"  :   [256, 256, 256, 512, 512],
                    }

## Number of training epochs
epochs = { 'magnet': {'2dlshape':   12000,
                      "2dhole"  :   6000,
                      "3dbeam"  :   600,
                      "3dbreast":   2000},
             'cnn' : {"2dlshape":   12000,
                       "3dbeam" :   150}}
