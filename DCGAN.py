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
import pandas as pd
import sklearn.preprocessing

class cDCGAN:
    
    
    def __init__(self,image_shape=(28,28,1),z_size=100,batch_size=50,num_epochs=50,generator_learning_rate =.001,discriminator_learning_rate=.001,path="./models/model.ckpt"):
        
        self.image_shape=image_shape
        self.batch_size=batch_size
        self.num_epochs = num_epochs
        self.generator_learning_rate=generator_learning_rate
        self.discriminator_learning_rate=discriminator_learning_rate
        self.model_path = path
        self.z_size=z_size
        #self.x_data,self.y_data = get_MNIST_data()
        
        ''' 
                                                reading the data from text files 
        '''
        
        data = pd.read_csv('Human-Emotion-Analysis-using-EEG-from-DEAP-dataset-master/dwt analysis/testFile_01.txt');
        required = data[['energy','entropy','sd','valence']]
        self.x_data = np.array(required)
        self.x_data = self.x_data.reshape((int)(self.x_data.shape[0]/10), self.x_data.shape[1]*10)

        self.y_data = np.array(data[['combined']])
        self.y_data = self.y_data.reshape((int)(self.y_data.shape[0]/10), self.y_data.shape[1]*10)
        '''
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(10))
        y_data = label_binarizer.transform(y_data)
        '''
        for i in range(31):
            if(i<8):
                data = pd.read_csv('Human-Emotion-Analysis-using-EEG-from-DEAP-dataset-master/dwt analysis/testFile_'+str(0)+str(i+2)+'.txt');
            else: 
                data = pd.read_csv('Human-Emotion-Analysis-using-EEG-from-DEAP-dataset-master/dwt analysis/testFile_'+str(i+2)+'.txt');
            required = data[['energy','entropy','sd','valence']]
            arr = np.array(required)
            arr = arr.reshape((int)(arr.shape[0]/10), arr.shape[1]*10)
            y_temp = np.array(data[['combined']])
            y_newD = y_temp.reshape((int)(y_temp.shape[0]/10), y_temp.shape[1]*10)
            self.x_data = np.concatenate((self.x_data,arr),axis=0)
            self.y_data = np.concatenate((self.y_data,y_newD),axis=0)
            
          
        print(self.x_data.shape)
        ''' 
                                                            preprocessing 
        '''
        
        self.scalar = sklearn.preprocessing.StandardScaler().fit(self.x_data)
        self.x_data = self.scalar.transform(self.x_data)
        
        self.scalar2 = sklearn.preprocessing.StandardScaler().fit(self.y_data)
        self.y_data = self.scalar2.transform(self.y_data)
        
        
        '''                                             tensorflow
        '''
        
        
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
            
            for batch in range(floor(self.x_data.shape[0]/self.batch_size)):
            #for batch in range():
 
                g_acc = 0
                d_acc = 0
                
                x_batch,y_batch = get_next_batch(self.x_data,self.y_data,batch,self.batch_size,img_shape=(10,4,1))
                z_batch = np.random.normal(0,1,size=(self.batch_size,100))
                
                d_loss,_ = sess.run([self.D_loss,self.D_train],feed_dict={self.x_ph:x_batch,self.y_ph:y_batch,self.z_ph:z_batch})
                g_loss,_ = sess.run([self.G_loss,self.G_train],feed_dict={self.z_ph:z_batch,self.y_ph:y_batch})
                
                
                show_data(epoch,batch,g_loss,d_loss,g_acc,d_acc)
                
                if batch==floor(self.x_data.shape[0]/self.batch_size)-1 :
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
        
        op_file= open("output.txt","w+")
            
        saver.restore(sess,self.model_path)
        #saver.restore(sess,self.scalar)
        #saver.restore(sess,self.scalar2)
        
        for i in range(10):
            
            y_input = np.zeros((1,10))
            y_input[0][i] = 1
            
            for j in range(5):
                
                z_input = np.random.normal(0,1,size=(1,100))
                output = sess.run(self.G,feed_dict={self.z_ph:z_input, self.y_ph:y_input})
                arr_img[i,j].imshow(output.reshape((10,4)))
                output = output.reshape((10,4))
                for k in range(10):
                    op_file.write("%2d,%.5f,%.5f,%.5f,%.5f\n"%(i*5+j,output[k,0],output[k,1],output[k,2],output[k,3]))   
             

        
        plt.show()
                    
                    
            
            
                    
        
                    
                
            
                
    
        