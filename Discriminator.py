#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 23:31:25 2019

@author: biswajit
"""
import tensorflow as tf

def Discriminator(x,y,reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope('dis', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.leaky_relu(x)
        #x = tf.layers.batch_normalization(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.leaky_relu(x)
        #x = tf.layers.batch_normalization(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        
        x = tf.layers.dense(x, 1024)
        x = tf.nn.leaky_relu(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 1)
        x = tf.nn.sigmoid(x)
        return x