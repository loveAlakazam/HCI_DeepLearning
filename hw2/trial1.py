#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 07:02:17 2018

@author: ailab
"""

import tensorflow as tf
import numpy as np
from cifar_10_load import load_cifar_gray_data_and_one_labels
def normalize_tensor(x): #normalization (0<vaule<1)
        max = tf.reduce_max(x)
        if(max==0):
            print('Warning Tensor is 0')
            return (x + 0.0000001)
        else:
            return (x/max)

def next_batch(x_data, y_data, batch_size):
    if (len(x_data) != len(y_data)):
        print('wrong data')
        return None, None
    batch_mask = np.random.choice(len(x_data), batch_size)
    x_batch = x_data[batch_mask]
    y_batch = y_data[batch_mask]
    return x_batch, y_batch
        
    
#load cifar gray data(images) and labels
gray_imgs, gray_labels =load_cifar_gray_data_and_one_labels('cifar_grey_data_and_labels.pkl')

#train 10000 datas 
#trian data set
train_imgs= gray_imgs[:7000, :]  #(7000,1024)
train_labels= gray_labels[:7000, :] #(7000, 10)

#test data set
test_imgs= gray_imgs[3000:, :]#(3000,1024)
test_labels= gray_labels[3000:, :]#(3000,10)

#convolution
X=tf.placeholder(tf.float32, [None,32,32,1]) #(None, 32*32)
Y=tf.placeholder(tf.float32, [None,10])

# conv-relu-pool layer1
#kernel_num=32
#kernel_num2=kernel_num1*2

W1= tf.Variable(tf.random_normal([5,5,1,32], stddev=0.01))
h_conv1 = tf.nn.conv2d( X, W1, strides=[1,1,1,1],padding='SAME') #(None,32,32,32)
h_conv_relu1=tf.nn.relu(h_conv1)
h_pool1 =tf.nn.max_pool(h_conv_relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #(None,16,16,32)

# conv-relu-pool layer2
W2= tf.Variable(tf.random_normal([5,5,32, 64], stddev=0.01))
h_conv2= tf.nn.conv2d( h_pool1, W2, strides=[1,1,1,1], padding='SAME') #(None, 16, 16, 64)
h_conv_relu2=tf.nn.relu(h_conv2)
h_pool2= tf.nn.max_pool(h_conv_relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')#(None,8,8,64)

# relu - dropout
W3= tf.Variable(tf.random_normal([8*8*64, 1024], stddev=0.01))
h_pool2_flat= tf.reshape(h_pool2, [-1, 8*8*64])
dropout_input= tf.nn.relu( tf.matmul(h_pool2_flat,W3)) #(?,1024)
keep_prob= tf.placeholder(tf.float32)
dropouted= tf.nn.dropout(dropout_input, keep_prob) #keep overfitting

# define hypothesis
W4=tf.Variable(tf.random_normal([1024,10], stddev=0.01))
b4= tf.Variable(tf.random_normal([10]), name='bias')
hypothesis=tf.matmul(dropouted,W4)+b4
##################################################################################
#define cost
cost= tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
#define trian -(to get minimize cost)
train = tf.train.AdamOptimizer(1e-4).minimize(cost)

#compute accuracy
predicted = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

#run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        batch_imgs, batch_labels= next_batch(train_imgs,
                                         train_labels, 
                                         batch_size=100)
    
        _, cost_val, train_accuracy= sess.run([train, cost, accuracy], 
                                   feed_dict={
                                           X:batch_imgs.reshape(-1,32,32,1), 
                                           Y:batch_labels, 
                                           keep_prob:0.5})
        if step%1000==0:
            print('train step{}: cost: {},  accuacy: {:.3f}'.format(step, cost_val,train_accuracy))
            
    #after train is finished
    print('Result Accuracy: ', sess.run(accuracy,feed_dict={X:test_imgs.reshape(-1,32,32,1),
                                                            Y:test_labels,
                                                            keep_prob:1.0}))
        
