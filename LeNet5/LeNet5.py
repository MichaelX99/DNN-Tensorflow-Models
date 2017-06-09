#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:26:34 2017

@author: mike
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.utils import shuffle

train_dir = '/home/mike/Deep_Learning_DataSets/mnist/train/'
test_dir = '/home/mike/Deep_Learning_DataSets/mnist/test/'

fpaths_train = []
fpaths_test = []
y_train = []
y_test = []
for i in range(10):
    temp = glob.glob(train_dir+str(i)+'/*.png')
    fpaths_train += temp
    y_train += len(temp)*[i]
    
    temp = glob.glob(test_dir+str(i)+'/*.png')
    fpaths_test += temp
    y_test += len(temp)*[i]

print('Importing training data')
X_train = []
for i in range(len(fpaths_train)):
    temp = cv2.imread(fpaths_train[i])
    X_train.append(temp)
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_train, y_train = shuffle(X_train, y_train)
   
print('Importing testing data')
X_test = []
for i in range(len(fpaths_test)):
    temp = cv2.imread(fpaths_test[i])
    X_test.append(temp)
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test, y_test = shuffle(X_test, y_test)

print('Defining network parameters')
# Define the training data tensor
tf_input = tf.placeholder(tf.float32, (None, 32, 32, 3))

# Define the target tensor
tf_target = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(tf_target, 10)

# Define the parameters of the network
sigma = .01
mu = 0
w1 = tf.Variable(tf.truncated_normal((5,5,3,6), mu, sigma))
w2 = tf.Variable(tf.truncated_normal((5,5,6,16), mu, sigma))
w3 = tf.Variable(tf.truncated_normal((400,120), mu, sigma))
w4 = tf.Variable(tf.truncated_normal((120,84), mu, sigma))
w5 = tf.Variable(tf.truncated_normal((84,10), mu, sigma))

b1 = tf.Variable(tf.zeros(6))
b2 = tf.Variable(tf.ones(16))
b3 = tf.Variable(tf.zeros(120))
b4 = tf.Variable(tf.ones(84))
b5 = tf.Variable(tf.ones(10))

# Define network
def LeNet5(x):
    # Layer 1 (Convolutional)
    out = tf.nn.conv2d(x, w1, [1,1,1,1], 'VALID') + b1
    out = tf.nn.sigmoid(out)
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1],'VALID')
    
    # Layer 2 (Convolutional)
    out = tf.nn.conv2d(out, w2, [1,1,1,1], 'VALID') + b2
    out = tf.nn.sigmoid(out)
    out = tf.nn.max_pool(out, [1,2,2,1], [1,2,2,1], 'VALID')
    
    # Layer 3 (Convolutionalish which becomes Fully Connected)
    out = tf.contrib.layers.flatten(out)
    out = tf.matmul(out, w3) + b3
    out = tf.nn.sigmoid(out)
    
    # Layer 4 (Fully Connected)
    out = tf.matmul(out, w4) + b4
    out = tf.nn.sigmoid(out)
    
    # Layer 5 (Fully Connected)
    out = tf.matmul(out, w5) + b5
    
    return out
logits = LeNet5(tf_input)

# Define loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

# Define optimizer
rate = 0.001
optimizer = tf.train.AdamOptimizer(rate)

# Define the training operation
training_operation = optimizer.minimize(loss_operation)

# Define evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={tf_input: batch_x, tf_target: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Define training hyperparameters
max_epochs = 100
batch_size = 256

# Initialize lists for plotting
validation = []
training = []
t = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("Training")
    for i in range(max_epochs):
        for offset in range(0, len(fpaths_train), batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={tf_input: batch_x, tf_target: batch_y})
        
        # Evaluate the network
        validation_accuracy = evaluate(X_test, y_test, batch_size)
        training_accuracy = evaluate(X_train, y_train, batch_size)
        
        # Append values to graph later
        validation.append(validation_accuracy)
        training.append(training_accuracy)
        t.append(i)
        
        # Print training status
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
# Visualize the learning
plt.plot(t, validation,t,training)
plt.legend(('Validation','Training'))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning')