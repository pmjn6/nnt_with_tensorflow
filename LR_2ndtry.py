# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:56:34 2016

@author: pejman
"""
import tensorflow as tf
import numpy as np
#data
sample_nbr = 100
#create two features:
x1 = np.random.uniform(-2,5,sample_nbr)
x2 = np.random.uniform(10,15,sample_nbr)
x  =np.array([x1,x2]).T

#Creat the target value as -x1+2*x2+2+noise
a = np.array([-1,2]).reshape(2,1)
y = np.matmul(x,a)+np.array([[2]])+np.random.random(sample_nbr).reshape((sample_nbr,1))




#variables. Since the data are feed to the model via 
#places hodder, the optimizer knows that it should only 
#minimize the cost by varying W and b
X = tf.placeholder(dtype=tf.float64,shape=[None,2])
Y = tf.placeholder(dtype=tf.float64,shape=[None,1])

W=tf.Variable(np.random.random(2).reshape((2,1)))
b = tf.Variable(np.array([[np.random.random()]]))

##model and cost of it
model = tf.add(tf.matmul(X,W),b)
cost =  tf.reduce_mean(input_tensor=tf.square(tf.sub(model,Y)),reduction_indices=0)

optimizer = tf.train.GradientDescentOptimizer(.0001).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    MINIMIZERS=[]
    sess.run(init)
    counter = 0
    for (e,f) in zip(x,y):
        sess.run(optimizer,feed_dict={X:e.reshape((1,2)), Y:f.reshape((1,1))})
        c,w=counter,sess.run([cost[0],W[0,:][0],W[1,:][0],b[0,:][0]],\
        feed_dict={X:e.reshape((1,2)), Y:f.reshape((1,1))})
        MINIMIZERS.append(w)
        if counter%5==0:
            print(c,"cost=",w[0],"a1=",w[1],"a2=",w[2],"b=",w[3])
        counter+=1
    minindex = np.argmin(np.array(MINIMIZERS)[:,0])
    print("min cost = ",np.array(MINIMIZERS)[minindex,0])
    print("best a1=",np.array(MINIMIZERS)[minindex,1])
    print("best a2=",np.array(MINIMIZERS)[minindex,2])
    print("best b=",np.array(MINIMIZERS)[minindex,3])
