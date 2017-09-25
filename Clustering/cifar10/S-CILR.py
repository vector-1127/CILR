# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 08:45:35 2016

@author: test
"""

from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['IMAGE_DIM_ORDERING'] = 'tf'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.Session(config=config)


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense,MaxPooling2D,Convolution2D,Highway,Activation
from keras.layers import Dropout,Flatten,Input,BatchNormalization
from keras import backend as K
from keras.utils import np_utils
from keras.engine.topology import Layer,InputSpec
from keras.regularizers import l2, activity_l2,activity_l1
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.noise import GaussianNoise
from PIL import Image
from keras.models import model_from_json
import h5py
import scipy.io as sio
import os,sys
from my_ImageDataGenerator import *
from myMetrics import *

global datagen
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    channel_shift_range=0.05,
    horizontal_flip=True,
    rescale=0.975,
    zoom_range=[0.95,1.05]
)


batch_size = 100
nb_epoch = 8
img_channels,img_rows,img_cols = 3,32,32
nb_layer = 10
aug = 2
nb_classes = 10


file = h5py.File('/home/changjianlong/datasets/CIFAR10.h5')
X_train = file['X_train'][:]
y_true = file['y_train'][:]
y_true = y_true.astype('int64')
file.close()
X_train = np.transpose(X_train,axes=(0,2,3,1))

inp_img = Input(shape=(img_rows,img_cols,img_channels))
x = GaussianNoise(0.02)(inp_img)
x = Convolution2D(32, 3, 3, border_mode='same')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Convolution2D(32, 3, 3, border_mode='same')(x)
x = BatchNormalization(mode=2,axis = 1)(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Flatten()(x)
x = Dense(512,activation = 'relu',activity_regularizer=activity_l1(1e-6))(x)
encode = BatchNormalization(mode=2)(x)

pre_train_model = Model(inp_img,encode)
pre_train_model.compile(loss = 'mse',optimizer = 'rmsprop')

for e in range(nb_epoch):
    y_train = pre_train_model.predict(X_train,batch_size = batch_size)
    pre_train_model.fit_generator(datagen.flow(X_train, y_train,batch_size=256),
                                  samples_per_epoch=aug*X_train.shape[0], nb_epoch=2)

    from sklearn.cluster import KMeans
    feature = pre_train_model.predict(X_train)
    kmeans = KMeans(n_clusters=10, random_state=0,max_iter=30000).fit(feature)
    y_pred = kmeans.predict(feature)
    y_true.shape=-1,
    acc = ACC(y_true,y_pred)
    nmi = NMI(y_true,y_pred)
    ari = ARI(y_true,y_pred)
    print(acc,nmi,ari)

