#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:43:34 2019

@author: biswajit
"""

import os
import tensorflow as tf
from DCGAN import cDCGAN

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    tf.reset_default_graph()
    
    DCGAN = cDCGAN(num_epochs=10,discriminator_learning_rate=.0001,generator_learning_rate=.001,batch_size=50)
    DCGAN.train()
    DCGAN.test()