# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 03:51:20 2018

@author: loveAlakazam
"""
# 5 Layers
import tensorflow as tf
import numpy as np

x_data= np.array([[0,0], [0,1], [1,0], [1,1]])  #(4,2)
y_data= np.array([[0],[1],[1],[0]])             #(4,1)  

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W1= tf.Variable(tf.random_normal([2,10]), name='weight1')
b1= tf.Variable(tf.random_normal([10]), name='bias1')
layer1= tf.sigmoid(tf.matmul(X, W1)+ b1) #(4,2)*(2,10)=(4,10)

W2= tf.Variable(tf.random_normal([10,20]), name='weight2')
b2=tf.Variable(tf.random_normal([20]), name='bias2')
layer2= tf.sigmoid(tf.matmul(layer1,W2)+b2)#(4,10)*(10,20)=(4,20)

W3=tf.Variable(tf.random_normal([20,20]), name='weight3')
b3=tf.Variable(tf.random_normal([20]), name='bias3')
layer3= tf.sigmoid(tf.matmul(layer2,W3)+b3) #(4,20)*(20,20)=(4,20)

W4=tf.Variable(tf.random_normal([20,1]), name='weight4')
b4=tf.Variable(tf.random_normal([1]),name='bias4')
hypothesis=tf.sigmoid(tf.matmul(layer3,W4)+b4) #(4,20)*(20,1)=(4,1)

#cost
cost= -tf.reduce_mean( Y*tf.log(hypothesis)+ (1-Y)*tf.log(1-hypothesis) )

#train
train= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#accuracy 계산
predicted= tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy= tf.reduce_mean( tf.cast( tf.equal(predicted,Y),dtype=tf.float32))

#graph run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        loss,_=sess.run([cost, train], feed_dict=feed)
        sess.run([W1,W2,W3,W4])
        if step % 100 ==0:
            print('cost: ', loss)
    #accuracy 출력
    predict, hypo, accr= sess.run([predicted,hypothesis, accuracy], feed_dict=feed)
    print('predicted: ', predict,'\naccuracy: ', accr)
