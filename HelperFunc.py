#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 23:33:34 2019

@author: biswajit
"""

import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import sys

def loss_func(predicted,labels):
    return tf.reduce_mean(tf.square(predicted-labels))



def get_next_batch(x_data,y_data,itr,batch_size,img_shape=(28,28,1)):
    
    x_next_batch = x_data[itr*batch_size : (itr+1)*batch_size, : , :]
    x_next_batch = np.reshape(x_next_batch,(batch_size,img_shape[0],img_shape[1],img_shape[2]))
    y_next_batch = y_data[itr*batch_size : (itr+1)*batch_size, : ]
    
    return x_next_batch,y_next_batch



def get_MNIST_data():
    
    mnist = tf.keras.datasets.mnist
    
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    
    print(x_train.shape)
    
    x_data = np.concatenate((x_train,x_test),axis=0)
    y_data = np.concatenate((y_train,y_test))
    
    x_data = x_data/255
    
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(10))
    y_data = label_binarizer.transform(y_data)
    
    return x_data,y_data
    
    

def show_data(epoch,batch,g_loss,d_loss,g_acc,d_acc):
    sys.stdout.write("\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (epoch,batch,g_loss,d_loss,g_acc,d_acc))
    sys.stdout.flush()