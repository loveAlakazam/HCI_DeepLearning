# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 02:43:30 2018

@author: loveAlakazam
"""
import tensorflow as tf
# y_data= x1 ^ x2 
x_data= [ [0,0], [0,1], [1,0], [1,1]]   #x_data.shape(4,2)
y_data= [ [0],   [1],   [1],   [0]]     #y_data.shape(4,1)
# xor 연산 결과는 0과 1밖에 없으므로 logistic Regression 을 이용한다.
# logistic Regression에 사용되는 활성화 함수는 sigmoid 이다. 
# 이는 0~1사이를 나타내고 모든 결과값의 합이 1이다.
# 그리고 cost를 구하는 손실함수는  -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))이다.

# placeholder로 입력을 받는다. 나중에 session run할때 feed_dict에서 X와 Y를 받는다.
X=tf.placeholder(tf.float32) 
Y=tf.placeholder(tf.float32) 

#1층
W1=tf.Variable(tf.random_normal([2,20]), name='weight1')  #W1.shape=(2,20)
b1=tf.Variable(tf.random_normal([20]), name='bias1')      #b1.shape=(20,)   
layer1= tf.sigmoid(tf.matmul(X,W1)+b1)                    #layer1.shape(4,20)

#2층
W2=tf.Variable(tf.random_normal([20,1]), name='weight2') #W2.shape=(20,1)
b2=tf.Variable(tf.random_normal([1]), name='bias2')      #b2.shape=(1,)
hypothesis =tf.sigmoid(tf.matmul(layer1, W2)+b2)         #hypothesis.shape(4,1)

#cost : 손실값
cost= -tf.reduce_mean( Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

# train: cost가 최소인 방향으로 학습한다.
# 학습에 사용되는 함수는 GradientDescentOptimizer이다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#확률구하기
#hypothesis가 0.5보다 크면 1(True)로 정의한다.
# 반대로 0.5보다 작거나 같으면 0(False)로 정의한다.
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32) #예측

#확률구하기
#1. tf.equal( predicted, Y) : predicted와 Y가 서로 같은지 확인 : True/ False 출력
#2. tf.cast ( 조건, dtype) : 조건(tf.equal(predicted, Y)이 True이면 1의 값을, False면 0의 값을 출력
#3. tf.mean: 더하여 평균을 구한다. 
accuracy= tf.reduce_mean( tf.cast( tf.equal(predicted, Y), dtype=tf.float32 )) 

#run graph 그래프 실행시키기
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed= {X: x_data, Y: y_data}
    for step in range(10001):
        #학습(train)을 실행. cost와 hypothesis도 실행
        sess.run(train, feed_dict=feed) 
        if step%100==0:
            print('cost: ',sess.run(cost, feed_dict=feed))
            
    #학습이 종료되면 예측값과 확률을 출력한다.
    predict, hypo, accr = sess.run([predicted,hypothesis,accuracy], \
                                   feed_dict=feed)
    print('hypothesis:\n', hypo,'\npredicted:\n',predict ,'\naccuracy: ',accr)
    
