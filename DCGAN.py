#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:38:07 2019

@author: biswajit
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:48:07 2019

@author: biswajit
"""


import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
from HelperFunc import loss_func
from HelperFunc import get_next_batch
import numpy as np
from HelperFunc import get_MNIST_data
from HelperFunc import show_data
import matplotlib.pyplot as plt
from math import floor 

class cDCGAN:
    
    
    def __init__(self,image_shape=(28,28,1),z_size=100,batch_size=50,num_epochs=50,generator_learning_rate =.001,discriminator_learning_rate=.001,num_images=70000,path="./models/model.ckpt"):
        
        self.image_shape=image_shape
        self.batch_size=batch_size
        self.num_epochs = num_epochs
        self.generator_learning_rate=generator_learning_rate
        self.discriminator_learning_rate=discriminator_learning_rate
        self.num_images = num_images
        self.model_path = path
        self.z_size=z_size
        self.x_data,self.y_data = get_MNIST_data()
        
        self.x_ph = tf.placeholder(tf.float32,shape=[None,self.image_shape[0],self.image_shape[1],self.image_shape[2]])
        self.y_ph = tf.placeholder(tf.float32,shape=[None,10])
        self.z_ph = tf.placeholder(tf.float32,shape=[None,self.z_size])
        
        self.G = Generator(self.z_ph,self.y_ph)
        D_output_real = Discriminator(self.x_ph,self.y_ph)
        D_output_fake = Discriminator(self.G,self.y_ph,reuse=True)
        
        D_real = loss_func(tf.ones_like(D_output_real)*.9,D_output_real)
        D_fake = loss_func(tf.zeros_like(D_output_fake),D_output_fake)
        
        self.D_loss = tf.add(D_real,D_fake)
        self.G_loss = loss_func(tf.ones_like(D_output_fake),D_output_fake)
        
        tvars = tf.trainable_variables()
        
        self.gvars = [var for var in tvars if "gen" in var.name]
        self.dvars = [var for var in tvars if "dis" in var.name]

        self.G_train = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate).minimize(self.G_loss,var_list=self.gvars)
        self.D_train = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate).minimize(self.D_loss,var_list=self.dvars)
        
        D_real_accuracy = tf.metrics.accuracy(tf.ones_like(D_output_real),D_output_real)
        D_fake_accuracy = tf.metrics.accuracy(tf.zeros_like(D_output_fake),D_output_fake)
        
        self.D_accuracy = (D_real_accuracy + D_fake_accuracy)
        self.G_accuracy = tf.metrics.accuracy(tf.ones_like(D_output_fake),D_output_fake)
        
        
    def train(self):
        
        
        
        init = tf.global_variables_initializer()
        
        cfg = tf.ConfigProto(allow_soft_placement=True )
        cfg.gpu_options.allow_growth = True

        sess = tf.Session(config=cfg)
        sess.run(init)

        saver = tf.train.Saver(var_list=self.gvars)
        
        for epoch in range(self.num_epochs):
            
            for batch in range(floor(self.num_images/self.batch_size)):
            #for batch in range():
 
                g_acc = 0
                d_acc = 0
                
                x_batch,y_batch = get_next_batch(self.x_data,self.y_data,batch,self.batch_size)
                z_batch = np.random.normal(0,1,size=(self.batch_size,100))
                
                d_loss,_ = sess.run([self.D_loss,self.D_train],feed_dict={self.x_ph:x_batch,self.y_ph:y_batch,self.z_ph:z_batch})
                g_loss,_ = sess.run([self.G_loss,self.G_train],feed_dict={self.z_ph:z_batch,self.y_ph:y_batch})
                
                
                show_data(epoch,batch,g_loss,d_loss,g_acc,d_acc)
                
                if batch==floor(self.num_images/self.batch_size)-1 :
                    print()
                    #print("epoch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" % (epoch,g_loss,d_loss,g_acc,d_acc))
                    
        save_path = saver.save(sess,self.model_path)
        print("Model saved in path: %s" % save_path)

                    
    def test(self):
        
        init = tf.global_variables_initializer()

        cfg = tf.ConfigProto(allow_soft_placement=True )
        cfg.gpu_options.allow_growth = True
        sess = tf.Session(config=cfg)
        sess.run(init)

        saver = tf.train.Saver(var_list=self.gvars)
        
        f,arr_img = plt.subplots(10,5)
        
 
            
        saver.restore(sess,self.model_path)
        
        for i in range(10):
            
            y_input = np.zeros((1,10))
            y_input[0][i] = 1
            
            for j in range(5):
                
                z_input = np.random.normal(0,1,size=(1,100))
                output = sess.run(self.G,feed_dict={self.z_ph:z_input, self.y_ph:y_input})
                arr_img[i,j].imshow(output.reshape((28,28)))

        
        plt.show()
                    
                    
            
            
                    
        
                    
                
            
                
    
        