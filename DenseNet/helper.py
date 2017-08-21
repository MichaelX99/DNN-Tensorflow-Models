import scipy.io as sio
import glob
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import numpy as np
import os

IMAGENET = '/home/mikep/hdd/DataSets/ImageNet2012/'
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
N_GPUS = 3
BATCH_SIZE = 64 * N_GPUS
IMAGE_SIZE = 224
MAX_EPOCH = 10
NUM_CLASSES = 1000 + 1
DECAY = 0.0001
TRAIN_SHARDS = 128
VALIDATION_SHARDS = 24
NUM_THREADS = 8
MAX_STEPS = 10000000

def variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def convolution(images, shape, decay, stride, in_scope):
    with tf.variable_scope(in_scope) as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=shape,
                                            stddev=5e-2,
                                            wd=decay)

        conv = tf.nn.conv2d(images, kernel, [1, stride, stride, 1], padding='SAME')
        biases = variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        batch_norm = tf.contrib.layers.batch_norm(conv)

    return batch_norm

def transition(input, k, b, scope):
    transition = convolution(input, [1,1,k*b,k], DECAY, 1, scope)
    transition = tf.nn.max_pool(transition, [1,3,3,1], [1,2,2,1], 'SAME')

    return transition

def dense_block(input, k, b, scope):
    conv = convolution(input, [1,1,k,k], DECAY, 1, scope+'_1')
    conv = convolution(conv, [3,3,k,k], DECAY, 1, scope+'_2')
    for i in range(b - 1):
        l = convolution(conv, [1,1,(i+1)*k,k], DECAY, 1, scope + '_' + str(i+2) + '_1')

        l = convolution(l, [3,3,k,k], DECAY, 1, scope + '_' + str(i+2) + '_2')

        conv = tf.concat([conv, l], axis=3)

    return conv
