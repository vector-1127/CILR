# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 08:45:35 2016

@author: test
"""

from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['IMAGE_DIM_ORDERING'] = 'tf'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf.Session(config=config)


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
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
    rotation_range=20,
    width_shift_range=0.18,
    height_shift_range=0.18,
    zoom_range=[0.85,1.15]
)

batch_size = 100
nb_epoch = 4
img_channels,img_rows,img_cols = 1,28,28
nb_layer = 10
aug = 8
nb_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, img_channels,img_rows,img_cols)
X_test = X_test.reshape(10000, img_channels*img_rows*img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_true = y_train

inp_img = Input(shape=(img_channels,img_rows,img_cols,))
x = GaussianNoise(0.2)(inp_img)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu',W_regularizer=l2(0.00001))(x)
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
acc = ACC(y_true,y_pred)
nmi = NMI(y_true,y_pred)
ari = ARI(y_true,y_pred)
print(acc,nmi,ari)

