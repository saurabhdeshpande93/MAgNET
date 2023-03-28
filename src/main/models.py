
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from layers.mag import  MAg
from layers.g_pool import G_Pool
from layers.g_unpool import G_Unpool
from utils import nth_adjacency
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, ReLU, LeakyReLU, BatchNormalization, concatenate, Lambda
from keras.layers import MaxPooling2D, MaxPooling3D, Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, UpSampling3D
import tensorflow as tf

tf.random.set_seed(50)


class Models():

  def __init__(self, input_shape, channels):
      self.input_shape = input_shape
      self.channels = channels


  def mag(self, channels, adjacency, hop, **kwargs):
      activation = LeakyReLU(alpha=0.01) if 'activation' not in kwargs else kwargs.get('activation')
      return MAg(channels=channels, activation=activation, adjacency= nth_adjacency(adjacency,hop))


  def conv2d(self, channels, input, **kwargs):
      activation = ReLU() if 'activation' not in kwargs else kwargs.get('activation')
      kernel = (3,3) if 'kernel' not in kwargs else kwargs.get('kernel')

      conv = Conv2D(channels, kernel, padding='same', data_format='channels_first')(input)
      if activation is not None:
         conv = BatchNormalization()(conv)
         conv = activation(conv)

      return conv


  def conv3d(self, channels,input, **kwargs):
      activation = ReLU() if 'activation' not in kwargs else kwargs.get('activation')
      kernel = (3,3,3) if 'kernel' not in kwargs else kwargs.get('kernel')

      conv = Conv3D(channels, kernel, padding='same', data_format='channels_first')(input)
      if activation is not None:
         conv = BatchNormalization()(conv)
         conv = activation(conv)

      return conv


  def magnet_2d(self, pooled_adjacencies, subgraphs):

      c0, c1, c2, c3 = self.channels
      A, Ap1, Ap2, Ap3 = pooled_adjacencies
      subgraph1, subgraph2, subgraph3 = subgraphs

      inputs = Input(shape=self.input_shape)

      mag1  =  self.mag(c0, A, 2)(inputs)
      mag1  =  self.mag(c0, A, 2)(mag1)
      pool1 =  G_Pool(subgraph1, nodes=len(A))(mag1)

      mag2  =  self.mag(c1, Ap1, 2)(pool1)
      mag2  =  self.mag(c1, Ap1, 2)(mag2)
      pool2 =  G_Pool(subgraph2, nodes=len(Ap1))(mag2)

      mag3  =  self.mag(c2, Ap2, 2)(pool2)
      mag3  =  self.mag(c2, Ap2, 2)(mag3)
      pool3 =  G_Pool(subgraph3, nodes=len(Ap2))(mag3)

      mag4  =  self.mag(c3, Ap3, 2)(pool3)
      mag4  =  self.mag(c3, Ap3, 2)(mag4)

      up1   =  G_Unpool(subgraph3, nodes=len(Ap2))(mag4)
      up1   =  tf.concat((mag3,up1),axis=1)
      mag5  =  self.mag(c2, Ap2, 2)(up1)
      mag5  =  self.mag(c2, Ap2, 2)(mag5)

      up2   =  G_Unpool(subgraph2, nodes=len(Ap1))(mag5)
      up2   =  tf.concat((mag2,up2),axis=1)
      mag6  =  self.mag(c1, Ap1, 2)(up2)
      mag6  =  self.mag(c1, Ap1, 2)(mag6)

      up3   =  G_Unpool(subgraph1, nodes=len(A))(mag6)
      up3   =  tf.concat((mag1,up3),axis=1)
      mag7  =  self.mag(c0, A, 2)(up3)
      mag7  =  self.mag(c0, A, 2)(mag7)

      out   =  self.mag(2, A, 2, activation=None)(mag7)

      model =  Model(inputs=inputs, outputs=out)

      return model


  def magnet_3dbeam(self, pooled_adjacencies, subgraphs):

      c0, c1, c2, c3, c4, c5 = self.channels
      A, Ap1, Ap2, Ap3, Ap4, Ap5 = pooled_adjacencies
      subgraph1, subgraph2, subgraph3, subgraph4, subgraph5 = subgraphs

      inputs = Input(shape=self.input_shape)

      mag1  =  self.mag(c0, A, 2)(inputs)
      pool1 =  G_Pool(subgraph1, nodes=len(A))(mag1)

      mag2  =  self.mag(c1, Ap1, 2)(pool1)
      pool2 =  G_Pool(subgraph2, nodes=len(Ap1))(mag2)

      mag3  =  self.mag(c2, Ap2, 2)(pool2)
      pool3 =  G_Pool(subgraph3, nodes=len(Ap2))(mag3)

      mag4  =  self.mag(c3, Ap3, 2)(pool3)
      pool4 =  G_Pool(subgraph4, nodes=len(Ap3))(mag4)

      mag5  =  self.mag(c4, Ap4, 2)(pool4)
      pool5 =  G_Pool(subgraph5, nodes=len(Ap4))(mag5)

      mag6  =  self.mag(c5, Ap5, 3)(pool5)

      up1   =  G_Unpool(subgraph5, nodes=len(Ap4))(mag6)
      up1   =  tf.concat((mag5,up1),axis=1)
      mag7  =  self.mag(c4, Ap4, 2)(up1)

      up2   =  G_Unpool(subgraph4, nodes=len(Ap3))(mag7)
      up2   =  tf.concat((mag4,up2),axis=1)
      mag8  =  self.mag(c3, Ap3, 2)(up2)

      up3   =  G_Unpool(subgraph3, nodes=len(Ap2))(mag8)
      up3   =  tf.concat((mag3,up3),axis=1)
      mag9  =  self.mag(c2, Ap2, 2)(up3)

      up4   =  G_Unpool(subgraph2, nodes=len(Ap1))(mag9)
      up4   =  tf.concat((mag2,up4),axis=1)
      mag10  =  self.mag(c1, Ap1, 2)(up4)

      up5   =  G_Unpool(subgraph1, nodes=len(A))(mag10)
      up5   =  tf.concat((mag1,up5),axis=1)
      mag11  =  self.mag(c0, A, 2)(up5)

      out   =  self.mag(3, A, 3, activation=None)(mag11)

      model =  Model(inputs=inputs,outputs=out)

      return model


  def magnet_3dbreast(self, pooled_adjacencies, subgraphs):

        c0, c1, c2, c3, c4 = self.channels
        A, Ap1, Ap2, Ap3, Ap4 = pooled_adjacencies
        subgraph1, subgraph2, subgraph3, subgraph4 = subgraphs

        inputs = Input(shape=self.input_shape)

        mag1  =  self.mag(c0, A, 2)(inputs)
        pool1 =  G_Pool(subgraph1, nodes=len(A))(mag1)

        mag2  =  self.mag(c1, Ap1, 2)(pool1)
        pool2 =  G_Pool(subgraph2, nodes=len(Ap1))(mag2)

        mag3  =  self.mag(c2, Ap2, 2)(pool2)
        pool3 =  G_Pool(subgraph3, nodes=len(Ap2))(mag3)

        mag4  =  self.mag(c3, Ap3, 2)(pool3)
        pool4 =  G_Pool(subgraph4, nodes=len(Ap3))(mag4)

        mag5  =  self.mag(c4, Ap4, 3)(pool4)

        up1   =  G_Unpool(subgraph4, nodes=len(Ap3))(mag5)
        up1   =  tf.concat((mag4,up1),axis=1)
        mag6  =  self.mag(c3, Ap3, 2)(up1)

        up2   =  G_Unpool(subgraph3, nodes=len(Ap2))(mag6)
        up2   =  tf.concat((mag3,up2),axis=1)
        mag7  =  self.mag(c2, Ap2, 2)(up2)

        up3   =  G_Unpool(subgraph2, nodes=len(Ap1))(mag7)
        up3   =  tf.concat((mag2,up3),axis=1)
        mag8  =  self.mag(c1, Ap1, 2)(up3)

        up4   =  G_Unpool(subgraph1, nodes=len(A))(mag8)
        up4   =  tf.concat((mag1,up4),axis=1)
        mag9  =  self.mag(c0, A, 2)(up4)

        out   =  self.mag(3, A, 3, activation=None)(mag9)

        model =  Model(inputs=inputs,outputs=out)

        return model


  def cnn_2d(self):

      dim, n_x, n_y = self.input_shape
      c0, c1, c2 = self.channels

      inputs = Input(shape=(dim, n_x, n_y))

      conv1 = ZeroPadding2D(padding=2, data_format='channels_first')(inputs)
      conv1 = self.conv2d(c0,conv1)
      conv1 = self.conv2d(c0,conv1)
      pool1 = MaxPooling2D((2,2),data_format='channels_first')(conv1)

      conv2 = self.conv2d(c1, pool1)
      conv2 = self.conv2d(c1, conv2)
      pool2 = MaxPooling2D((2,2),data_format='channels_first')(conv2)

      conv3 = self.conv2d(c2, pool2)
      conv3 = self.conv2d(c2, conv3)

      up1 = UpSampling2D(size=(2,2),data_format='channels_first')(conv3)
      up1 = concatenate([conv2,up1],axis=1)
      conv4 = self.conv2d(c1, up1)
      conv4 = self.conv2d(c1, conv4)

      up2 = UpSampling2D(size=(2,2),data_format='channels_first')(conv4)
      up2 = concatenate([conv1,up2],axis=1)
      conv5 = self.conv2d(c0, up2)
      conv5 = self.conv2d(c0, conv5)

      conv6 = self.conv2d(dim, conv5, activation=None, kernel=(1,1))
      conv6 = Lambda(lambda x: x[:, :, 2:(n_x+2), 2:(n_y+2)])(conv6)

      model  = Model(inputs=inputs, outputs=conv6)

      return model


  def cnn_3d(self):

      dim, n_x, n_y , n_z =  self.input_shape
      c0, c1, c2, c3, c4 = self.channels

      inputs = Input(shape=(dim, n_x, n_y , n_z))
      ##
      conv1 = ZeroPadding3D(padding=2, data_format='channels_first')(inputs)
      conv1 = self.conv3d(c0, conv1)
      conv1 = self.conv3d(c0, conv1)
      pool1 = MaxPooling3D((2,2,2), data_format='channels_first')(conv1)
      ##
      conv2 = self.conv3d(c1, pool1)
      conv2 = self.conv3d(c1, conv2)
      pool2 = MaxPooling3D((2,2,2), data_format='channels_first')(conv2)
      ##
      conv3 = self.conv3d(c2, pool2)
      conv3 = self.conv3d(c2, conv3)
      pool3 = MaxPooling3D((2,2,2), data_format='channels_first')(conv3)
      ##
      conv4 = self.conv3d(c3,pool3)
      conv4 = self.conv3d(c3,conv4)
      pool4 = MaxPooling3D((2,2,2),data_format='channels_first')(conv4)
      ##
      conv5 = self.conv3d(c4,pool4)
      conv5 = self.conv3d(c4,conv5)
      ##
      up1 = UpSampling3D(size=(2,2,2),data_format='channels_first')(conv5)
      up1 = concatenate([conv4,up1],axis=1)
      conv6 = self.conv3d(c3,up1)
      conv6 = self.conv3d(c3,conv6)
      ##
      up2 = UpSampling3D(size=(2,2,2),data_format='channels_first')(conv6)
      up2 = concatenate([conv3,up2],axis=1)
      conv7 = self.conv3d(c2, up2)
      conv7 = self.conv3d(c2, conv7)
      #
      up3 = UpSampling3D(size=(2,2,2),data_format='channels_first')(conv7)
      up3 = concatenate([conv2,up3],axis=1)
      conv8 = self.conv3d(c1, up3)
      conv8 = self.conv3d(c1, conv8)
      #
      up4 = UpSampling3D(size=(2,2,2),data_format='channels_first')(conv8)
      up4 = concatenate([conv1,up4],axis=1)
      conv9 = self.conv3d(c0, up4)
      conv9 = self.conv3d(c0, conv9)

      conv10 = self.conv3d(dim, conv9, activation=None, kernel=(1,1,1))
      conv10 = Lambda(lambda x: x[:,:,2:(n_x+2), 2:(n_y+2),2:(n_z+2)])(conv10)

      model = Model(inputs=inputs, outputs=conv10)

      return model

  def train(self, model, data, path, lr_scheduler, epochs):

      X_train, Y_train, X_test, Y_test = data

      opt = Adam(lr = 0.0001)
      model.compile(optimizer=opt, loss='mean_squared_error')

      checkpoint = [LearningRateScheduler(lr_scheduler, verbose=1), ModelCheckpoint(path, monitor='val_loss', verbose=1,
      save_best_only=True,save_weights_only=True, mode='min', period=1)]

      print("\n \n === Training from scratch === \n \n ")

      hist = model.fit(X_train, Y_train,
                      epochs= epochs,
                      batch_size = 4,
                      shuffle=False, validation_data=(X_test , Y_test),
                      callbacks=checkpoint)

      return print(" \n \n === Training procedure completed === \n \n")

  @staticmethod
  def print_summary(model, print=False):
      if print == True:
          return model.summary()
