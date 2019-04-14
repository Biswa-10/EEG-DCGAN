#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:57:44 2019

@author: biswajit
"""
import tensorflow as tf

def Generator(z,y,reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        z = tf.layers.dense(z, units=3 * 3 * 256)
        z = tf.nn.leaky_relu(z)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        z = tf.reshape(z, shape=(-1, 3, 3, 256))
        z = tf.layers.conv2d_transpose(z,128,2,strides=2)
        z = tf.layers.batch_normalization(z)
        # Deconvolution, image shape: (batch, 14, 14, 64)
        z = tf.layers.conv2d_transpose(z, 256, 4, strides=2)
        z = tf.layers.batch_normalization(z)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        z = tf.layers.conv2d_transpose(z, 1, 2, strides=2)
        #z = tf.layers.batch_normalization(z)
        # Apply sigmoid to clip values between 0 and 1
        z = tf.nn.sigmoid(z)
        
        return z