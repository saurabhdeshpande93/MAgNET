import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocess():

    def __init__(self,df,dim,dof,**kwargs):
        self.df = df        #dataframe
        self.dim = dim      #Dimensionality of the problem 2D or 3D
        self.dof = dof      #Degree of freedom of the problem
        self.tot_n = int(len(self.df)/self.dof) #Total examples in the data
        self.n_nodes = int(self.dof/self.dim)   #Total number of nodes


    def get_data_UNET(self,split,mesh_dim):

        '''
        Create train test dataset for training CNNs from FEM dataset csv file

        Inputs

        - split: ratio for test set
        - mesh_dim: list containing number of FEM nodes in x,y (,z) directions [n_x, n_y] (,n_z])

        Outputs

        - X_train, X_test, Y_train, Y_test with shape as
        (n_train/n_test, n_ch, dim_x, dim_y) or  (n_train/n_test, n_ch, dim_x, dim_y, dim_z)
        depending on 2D or 3D problem

        '''

        X_org = np.zeros((self.tot_n, self.dof))
        Y_org = np.zeros((self.tot_n, self.dof))

        for i in range(self.tot_n):
            X_org[i, :] = self.df['f'].values[i * self.dof:(i + 1) * self.dof]
            Y_org[i, :] = self.df['u'].values[i * self.dof:(i + 1) * self.dof]

            X_org[i, :] = Preprocess.reorder(X_org[i, :],self.dof,self.dim)
            Y_org[i, :] = Preprocess.reorder(Y_org[i, :],self.dof,self.dim)


        X_train, X_test, Y_train, Y_test = train_test_split(X_org, Y_org, test_size=split,random_state=42)

        n_train = len(X_train)
        n_test = len(X_test)

        if self.dim==2:
            n_x, n_y = mesh_dim

            X_train = X_train.reshape(n_train,n_ch, dim_x, dim_y)
            Y_train = Y_train.reshape(n_train,n_ch, dim_x, dim_y)

            X_test = X_test.reshape(n_test, n_ch, dim_x, dim_y)
            Y_test = Y_test.reshape(n_test, n_ch, dim_x, dim_y)

        elif self.dim==3:
            n_x, n_y, n_z = mesh_dim
            n_ch, dim_x, dim_y, dim_z = self.dim, n_x, n_y, n_z

            X_train = X_train.reshape(n_train, n_ch, dim_x, dim_y, dim_z)
            Y_train = Y_train.reshape(n_train, n_ch, dim_x, dim_y, dim_z)

            X_test = X_test.reshape(n_test, n_ch, dim_x, dim_y, dim_z)
            Y_test = Y_test.reshape(n_test, n_ch, dim_x, dim_y, dim_z)


        return X_train, X_test, Y_train, Y_test


    def get_data_MAgNET(self,split):

         '''
         To generate data for MAgNET/Perceiver IO

         Returns:

         Training and test sets with shape as (n_train(& n_test),dof) with reordered dofs

         '''

         X_org = np.zeros((self.tot_n, self.dof))  # Features
         Y_org = np.zeros((self.tot_n, self.dof))

         for i in range(self.tot_n):
             X_org[i, :] = self.df['f'].values[i * self.dof:(i + 1) * self.dof]
             Y_org[i, :] = self.df['u'].values[i * self.dof:(i + 1) * self.dof]

             X_org[i,:] = Preprocess.reorder(X_org[i,:],self.dof,self.dim)
             Y_org[i,:] = Preprocess.reorder(Y_org[i,:],self.dof,self.dim)

         X_train, X_test, Y_train, Y_test = train_test_split(X_org, Y_org, test_size=split,random_state=42)


         return X_train, X_test, Y_train, Y_test



    @staticmethod
    def reorder(X,dof,dim):

        '''
        Original FEM data is stored as (1x, 1y, 2x, 2y ...,nx,ny)
        But for efficient use of MAgNET/CNNs, we need to re-arrange all x, y dofs in
        separate channels. 

        inputs:

        - Array of features and lables in the form  (1x, 1y, 1z, 2x, 2y, 2z ...,nx,ny,nz)

        outputs:

        - Reordered array in the form (1x, 2x, ... , nx, 1y, 2y ...ny ,1z, ....,nz)

        '''

        if dim == 2:
          dof_x = X[0:dof-1:dim]
          dof_y = X[1:dof:dim]

          reordered_dof = np.hstack((dof_x,dof_y))

        elif dim == 3:
          dof_x = X[0:dof-1:dim]
          dof_y = X[1:dof-1:dim]
          dof_z = X[2:dof:dim]

          reordered_dof = np.hstack((dof_x,dof_y,dof_z))

        return reordered_dof


    @staticmethod
    def org_order(reordered_dof,dof,dim):

        '''
        Get back the original dof arrangement

        Inputs:

        - Reordered array in the form (1x, 2x, ... , nx, 1y, 2y ...ny ,1z, ....,nz)

        Outputs:

        - Returns original ordering in the form  (1x, 1y, 1z, 2x, 2y, 2z ...,nx,ny,nz)

        '''

        original_dof = np.zeros((dof,))

        if dim==2:

          reordered_dof_x = reordered_dof[0:int(dof/dim)]
          reordered_dof_y = reordered_dof[int(dof/dim):dof]

          for i in range(int(dof/dim)):
             original_dof[2*i] = reordered_dof_x[i]
             original_dof[2*i+1] = reordered_dof_y[i]


        elif dim==3:

          reordered_dof_x = reordered_dof[0:int(dof/dim)]
          reordered_dof_y = reordered_dof[int(dof/dim):int(2*dof/dim)]
          reordered_dof_z = reordered_dof[int(2*dof/dim):dof]

          for i in range(int(dof/dim)):
             original_dof[3*i] = reordered_dof_x[i]
             original_dof[3*i+1] = reordered_dof_y[i]
             original_dof[3*i+2] = reordered_dof_z[i]

        return original_dof
