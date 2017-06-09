#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:29:40 2017

@author: mike
"""
#https://www.tensorflow.org/tutorials/deep_cnn#model_inputs
#https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py

import glob
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

# Define the path to the cifar10 dataset on my machine
cifar10_fpath = "/home/mike/Deep_Learning_DataSets/cifar10/"
#cifar10_fpath = "/media/mike/Backup File/cifar10/"

# Define the classes in the dataset and the numeric values that I am assigning them
letter_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
number_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_classes = len(number_classes)

# Initialize the filepaths and the labels
train_fpaths = []
y_train = []
test_fpaths = []
y_test = []

# Load the filepaths into memory
i = 0
for classes in letter_classes:
    temp_train = glob.glob(cifar10_fpath+"train/%s/*.png"%classes)
    train_fpaths += temp_train
    temp_y = [i] * len(temp_train)
    y_train += temp_y


    temp_test = glob.glob(cifar10_fpath+"test/%s/*.png"%classes)
    test_fpaths += temp_test
    temp_y = [i] * len(temp_test)
    y_test += temp_y

    i += 1

train_fpaths, y_train = shuffle(train_fpaths, y_train)
test_fpaths, y_test = shuffle(test_fpaths, y_test)

# Define the training data tensor
#tf_input = tf.placeholder(tf.float32, (None, 256, 256, 3))
tf_input = tf.placeholder(tf.float32, (None, 32, 32, 3))
tf_adjusted = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_input)

# Define the target tensor
tf_target = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(tf_target, num_classes)

# Define dropout probability
keep_prob = tf.placeholder(tf.float32)

# Define the parameters of the network
sigma = .01
mu = 0

with tf.name_scope('Parameters'):
    w1 = tf.Variable(tf.truncated_normal((5,5,3,64), mu, sigma))
    tf.summary.histogram('W1', w1)
    w2 = tf.Variable(tf.truncated_normal((5,5,64,64), mu, sigma))
    tf.summary.histogram('W2', w2)
    w3 = tf.Variable(tf.truncated_normal((8*8*64,384), mu, sigma))
    tf.summary.histogram('W3', w3)
    w4 = tf.Variable(tf.truncated_normal((384,192), mu, sigma))
    tf.summary.histogram('W4', w4)
    w5 = tf.Variable(tf.truncated_normal((192,num_classes), mu, sigma))
    tf.summary.histogram('W5', w5)

    b1 = tf.Variable(tf.zeros(64))
    tf.summary.histogram('B1', b1)
    b2 = tf.Variable(tf.ones(64))
    tf.summary.histogram('B2', b2)
    b3 = tf.Variable(tf.zeros(384))
    tf.summary.histogram('B3', b3)
    b4 = tf.Variable(tf.ones(192))
    tf.summary.histogram('B4', b4)
    b5 = tf.Variable(tf.ones(num_classes))
    tf.summary.histogram('B5', b5)


def network(x):
    with tf.name_scope('Layer1'):
        conv1 = tf.nn.conv2d(x, w1, [1,1,1,1], 'SAME') + b1
        tf.summary.histogram('Conv1', conv1)
        relu1 = tf.nn.relu(conv1)
        tf.summary.histogram('Relu1', relu1)
        pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], 'SAME')
        tf.summary.histogram('Pool1', pool1)
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=2, alpha=10e-4, beta=.75)
        tf.summary.histogram('Lrn1', lrn1)

    with tf.name_scope('Layer2'):
        conv2 = tf.nn.conv2d(lrn1, w2, [1,1,1,1], 'SAME') + b2
        tf.summary.histogram('Conv2', conv2)
        relu2 = tf.nn.relu(conv2)
        tf.summary.histogram('Relu2', relu2)
        lrn2 = tf.nn.local_response_normalization(relu2, depth_radius=5, bias=2, alpha=10e-4, beta=.75)
        tf.summary.histogram('Lrn2', lrn2)
        pool2 = tf.nn.max_pool(lrn2, [1,3,3,1], [1,2,2,1], 'SAME')
        tf.summary.histogram('Pool2', pool2)

    with tf.name_scope('Layer3'):
        dense = tf.contrib.layers.flatten(pool2)
        tf.summary.histogram('Dense', dense)
        fc1 = tf.matmul(dense, w3) + b3
        tf.summary.histogram('Fc1', fc1)
        relu3 = tf.nn.relu(fc1)
        tf.summary.histogram('Relu3', relu3)
        drop1 = tf.nn.dropout(relu3, keep_prob)
        tf.summary.histogram('Drop1', drop1)

    with tf.name_scope('Layer4'):
        fc2 = tf.matmul(drop1, w4) + b4
        tf.summary.histogram('Fc2', fc1)
        relu4 = tf.nn.relu(fc2)
        tf.summary.histogram('Relu4', relu4)
        drop2 = tf.nn.dropout(relu4, keep_prob)
        tf.summary.histogram('Drop2', drop2)

    with tf.name_scope('Layer5'):
        fc3 = tf.matmul(drop2, w5) + b5
        tf.summary.histogram('Fc3', fc3)

    return fc3

# Define training hyperparameters
max_epochs = 100
batch_size = 128

logits = network(tf_input)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

loss_operation = tf.reduce_mean(cross_entropy)
tf.summary.scalar('Loss', loss_operation)

# Define optimizer
global_step = 0
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)

moment = .9
optimizer = tf.train.AdamOptimizer(starter_learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate, moment)

# Define the training operation
training_operation = optimizer.minimize(loss_operation)

# Define evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def data_gen(fpaths):
    data = []
    for path in fpaths:
        temp = cv2.imread(path)
        temp = temp[:,:,::-1]
        #temp = cv2.resize(temp, (256, 256))
        data.append(temp)
    return data

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x = data_gen(X_data[offset:end])
        batch_y = y_data[offset:end]
        accuracy = sess.run(accuracy_operation, feed_dict={tf_input: batch_x, tf_target: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Initialize lists for plotting
validation = []
training = []
t = []

tensor_summary = tf.summary.merge_all()

otrain = []
oval = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('train',sess.graph)

    print("Training")
    for i in range(10):
        for offset in range(0, len(train_fpaths), batch_size):
            end = offset + batch_size
            batch_x = data_gen(train_fpaths[offset:end])
            batch_y = y_train[offset:end]
            _, out_summary = sess.run([training_operation, tensor_summary], feed_dict={tf_input: batch_x, tf_target: batch_y, keep_prob: .5})
            #print("current number = %d and total = %d"%(offset, batch_size*3))

        # Evaluate the network
        validation_accuracy = evaluate(test_fpaths, y_test, batch_size)
        training_accuracy = evaluate(train_fpaths, y_train, batch_size)

        # Append values to graph later
        validation.append(validation_accuracy)
        training.append(training_accuracy)
        t.append(i)
        writer.add_summary(out_summary)

        # Print training status
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()

    #saver.save(sess, './cifar_alexnet/cifar_alexnet')
    print("Model saved")
"""
# Visualize the learning
plt.plot(t, validation,t,training)
plt.legend(('Validation','Training'))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning')
"""
