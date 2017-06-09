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
size = 32
tf_input = tf.placeholder(tf.float32, (None, size, size, 3))
tf_adjusted = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf_input)

# Define the target tensor
tf_target = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(tf_target, num_classes)
tf.summary.histogram('one_hot', one_hot_y)

# Define dropout probability
keep_prob = tf.placeholder(tf.float32)

# Define the parameters of the network
sigma = .01
mu = 0

with tf.name_scope('Parameters'):
    with tf.name_scope('Layer1'):
        with tf.name_scope('Weights'):
            #w1 = tf.Variable(tf.truncated_normal((11,11,3,96), mu, sigma))
            w1 = tf.Variable(tf.truncated_normal((5,5,3,96), mu, sigma))
            tf.summary.histogram('Weight', w1)
            tf.add_to_collection('regularization', w1)
        with tf.name_scope('Bias'):
            b1 = tf.Variable(tf.zeros(96))
            tf.summary.histogram('Bias', b1)
            tf.add_to_collection('regularization', b1)

    with tf.name_scope('Layer2'):
        with tf.name_scope('Weights'):
            w2 = tf.Variable(tf.truncated_normal((5,5,96,256), mu, sigma))
            tf.summary.histogram('Weight', w2)
            tf.add_to_collection('regularization', w2)
        with tf.name_scope('Bias'):
            b2 = tf.Variable(tf.ones(256))
            tf.summary.histogram('Bias', b2)
            tf.add_to_collection('regularization', b2)

    with tf.name_scope('Layer3'):
        with tf.name_scope('Weights'):
            w3 = tf.Variable(tf.truncated_normal((3,3,256,384), mu, sigma))
            tf.summary.histogram('Weight', w3)
            tf.add_to_collection('regularization', w3)
        with tf.name_scope('Bias'):
            b3 = tf.Variable(tf.zeros(384))
            tf.summary.histogram('Bias', b3)
            tf.add_to_collection('regularization', b3)

    with tf.name_scope('Layer4'):
        with tf.name_scope('Weights'):
            w4 = tf.Variable(tf.truncated_normal((3,3,384,384), mu, sigma))
            tf.summary.histogram('Weight', w4)
            tf.add_to_collection('regularization', w4)
        with tf.name_scope('Bias'):
            b4 = tf.Variable(tf.ones(384))
            tf.summary.histogram('Bias', b4)
            tf.add_to_collection('regularization', b4)

    with tf.name_scope('Layer5'):
        with tf.name_scope('Weights'):
            w5 = tf.Variable(tf.truncated_normal((3,3,384,256), mu, sigma))
            tf.summary.histogram('Weight', w5)
            tf.add_to_collection('regularization', w5)
        with tf.name_scope('Bias'):
            b5 = tf.Variable(tf.ones(256))
            tf.summary.histogram('Bias', b5)
            tf.add_to_collection('regularization', b5)

    with tf.name_scope('Layer6'):
        with tf.name_scope('Weights'):
            w6 = tf.Variable(tf.truncated_normal((8*8*256,4096), mu, sigma))
            tf.summary.histogram('Weight', w6)
            tf.add_to_collection('regularization', w6)
        with tf.name_scope('Bias'):
            b6 = tf.Variable(tf.ones(4096))
            tf.summary.histogram('Bias', b6)
            tf.add_to_collection('regularization', b6)

    with tf.name_scope('Layer7'):
        with tf.name_scope('Weights'):
            w7 = tf.Variable(tf.truncated_normal((4096, 4096), mu, sigma))
            tf.summary.histogram('Weight', w7)
            tf.add_to_collection('regularization', w7)
        with tf.name_scope('Bias'):
            b7 = tf.Variable(tf.ones(4096))
            tf.summary.histogram('Bias', b7)
            tf.add_to_collection('regularization', b7)

    with tf.name_scope('Layer8'):
        with tf.name_scope('Weights'):
            w8 = tf.Variable(tf.truncated_normal((4096,num_classes), mu, sigma))
            tf.summary.histogram('Weight', w8)
            tf.add_to_collection('regularization', w8)
        with tf.name_scope('Bias'):
            b8 = tf.Variable(tf.ones(num_classes))
            tf.summary.histogram('Bias', b8)
            tf.add_to_collection('regularization', b8)

def network(x):
    tf.get_default_graph()

    with tf.name_scope('Layer1'):
        with tf.name_scope('Convolution'):
            #conv1 = tf.nn.conv2d(x, w1, [1,4,4,1], 'SAME') + b1
            conv1 = tf.nn.conv2d(x, w1, [1,1,1,1], 'SAME') + b1
            tf.summary.histogram('Conv1', conv1)
        with tf.name_scope('Activation'):
            relu1 = tf.nn.relu(conv1)
            tf.summary.histogram('Relu1', relu1)
        with tf.name_scope('Pool'):
            pool1 = tf.nn.max_pool(relu1, [1,3,3,1], [1,2,2,1], 'SAME')
            tf.summary.histogram('Pool1', pool1)
        with tf.name_scope('LRN'):
            lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=5, bias=2, alpha=10e-4, beta=.75)
            tf.summary.histogram('Lrn1', lrn1)

    with tf.name_scope('Layer2'):
        with tf.name_scope('Convolution'):
            conv2 = tf.nn.conv2d(lrn1, w2, [1,1,1,1], 'SAME') + b2
            tf.summary.histogram('Conv2', conv2)
        with tf.name_scope('Activation'):
            relu2 = tf.nn.relu(conv2)
            tf.summary.histogram('Relu2', relu2)
        with tf.name_scope('LRN'):
            lrn2 = tf.nn.local_response_normalization(relu2, depth_radius=5, bias=2, alpha=10e-4, beta=.75)
            tf.summary.histogram('Lrn2', lrn2)
        with tf.name_scope('Pool'):
            pool2 = tf.nn.max_pool(lrn2, [1,3,3,1], [1,2,2,1], 'SAME')
            tf.summary.histogram('Pool2', pool2)

    with tf.name_scope('Layer3'):
        with tf.name_scope('Convolution'):
            conv3 = tf.nn.conv2d(pool2, w3, [1,1,1,1], 'SAME') + b3
            tf.summary.histogram('Conv3', conv3)
        with tf.name_scope('Activation'):
            relu3 = tf.nn.relu(conv3)
            tf.summary.histogram('Relu3', relu3)

    with tf.name_scope('Layer4'):
        with tf.name_scope('Convolution'):
            conv4 = tf.nn.conv2d(relu3, w4, [1,1,1,1], 'SAME') + b4
            tf.summary.histogram('Conv4', conv4)
        with tf.name_scope('Activation'):
            relu4 = tf.nn.relu(conv4)
            tf.summary.histogram('Relu4', relu4)

    with tf.name_scope('Layer5'):
        with tf.name_scope('Convolution'):
            conv5 = tf.nn.conv2d(relu4, w5, [1,1,1,1], 'SAME') + b5
            tf.summary.histogram('Conv5', conv5)
        with tf.name_scope('Activation'):
            relu5 = tf.nn.relu(conv5)
            tf.summary.histogram('Relu5', relu5)
    #pool5 = tf.nn.max_pool(relu5, [1,3,3,1], [1,2,2,1], 'VALID')

    with tf.name_scope('Layer6'):
        with tf.name_scope('Flatten'):
            dense = tf.contrib.layers.flatten(relu5)
            tf.summary.histogram('Dense', dense)
        with tf.name_scope('Fully_Connected'):
            fc1 = tf.matmul(dense, w6) + b6
            tf.summary.histogram('FC1', fc1)
        with tf.name_scope('Activation'):
            relu6 = tf.nn.relu(fc1)
            tf.summary.histogram('Relu6', relu6)
        with tf.name_scope('Dropout'):
            drop1 = tf.nn.dropout(relu6, keep_prob)
            tf.summary.histogram('Drop1', drop1)

    with tf.name_scope('Layer7'):
        with tf.name_scope('Fully_Connected'):
            fc2 = tf.matmul(drop1, w7) + b7
            tf.summary.histogram('FC2', fc2)
        with tf.name_scope('Activation'):
            relu7 = tf.nn.relu(fc2)
            tf.summary.histogram('Relu7', relu7)
        with tf.name_scope('Dropout'):
            drop2 = tf.nn.dropout(relu7, keep_prob)
            tf.summary.histogram('Drop2', drop2)

    with tf.name_scope('Layer8'):
        with tf.name_scope('Fully_Connected'):
            fc3 = tf.matmul(drop2, w8) + b8
            tf.summary.histogram('Fc3', fc3)

    return fc3

# Define training hyperparameters
max_epochs = 100
batch_size = 128

#logits = network(tf_adjusted)
logits = network(tf_input)

with tf.name_scope('Cross_Entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    tf.summary.histogram('Cross_Entropy', cross_entropy)

with tf.name_scope('l2_loss'):
    l2_loss = [tf.nn.l2_loss(w) for w in tf.get_collection('regularization')]
    tf.summary.histogram('l2_loss', l2_loss)

with tf.name_scope('weight_magnitude'):
    weight_magnitude = tf.add_n(l2_loss)
    tf.summary.histogram('weight_magnitude', weight_magnitude)

with tf.name_scope('cross_entropy_loss'):
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('Cross_entropy_loss', cross_entropy_loss)

with tf.name_scope('Loss'):
    loss_operation = cross_entropy_loss + l2_loss
    tf.summary.histogram('Loss', loss_operation)

# Define optimizer
global_step = 0
starter_learning_rate = 0.00001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
tf.summary.scalar('Learning_Rate', learning_rate)

moment = .9
optimizer = tf.train.AdamOptimizer(starter_learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate, moment)

# Define the training operation
training_operation = optimizer.minimize(loss_operation)

# Define evaluation
with tf.name_scope('Evaluation'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram('accuracy', accuracy_operation)

def data_gen(fpaths):
    data = []
    for path in fpaths:
        temp = cv2.imread(path)
        temp = temp[:,:,::-1]
        temp = cv2.resize(temp, (size, size))
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
    return total_accuracy / num_examples#, out

tensor_summary = tf.summary.merge_all()

# Initialize lists for plotting
validation = []
training = []
t = []


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('train',sess.graph)

    print("Training")
    start_time = time.clock()
    for i in range(5):
        for offset in range(0, len(train_fpaths), batch_size):
        #for offset in range(0, batch_size, batch_size):
            end = offset + batch_size
            batch_x = data_gen(train_fpaths[offset:end])
            batch_y = y_train[offset:end]
            _, out_summary = sess.run([training_operation, tensor_summary], feed_dict={tf_input: batch_x, tf_target: batch_y, keep_prob: .5})
            writer.add_summary(out_summary,i)

        # Evaluate the network
        validation_accuracy = evaluate(test_fpaths, y_test, batch_size)
        training_accuracy = evaluate(train_fpaths, y_train, batch_size)

        # Append values to graph later
        validation.append(validation_accuracy)
        training.append(training_accuracy)
        t.append(i)


        # Print training status
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()

    #saver.save(sess, './alexnet/alexnet')
    writer.close()
    print("Model saved")
    print('seconds elapsed = ',time.clock()-start_time)
"""
# Visualize the learning
plt.plot(t, validation,t,training)
plt.legend(('Validation','Training'))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning')
"""
