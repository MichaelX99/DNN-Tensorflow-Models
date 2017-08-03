import scipy.io as sio
import glob
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import numpy as np

IMAGENET = '/home/mikep/hdd/DataSets/ImageNet2012/'
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
N_GPUS = 2
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_EPOCH = 10
NUM_CLASSES = 1000
DECAY = 0.0001


def import_ImageNet(ImageNet_fpath):
    """Helper to import ImageNet

    Args:
      ImageNet_fpath: path to ImageNet

    Returns:
      train_fpaths: list of filepaths to every training example
      train_target: list of numeric class labels for the training examples
      valid_fpaths: list of filepaths to every validation example
      valid_target: list of numeric class labels for the validation examples
    """
    meta = sio.loadmat(ImageNet_fpath + 'DevKit/data/meta.mat')
    synsets = meta['synsets']

    synset_ids = []
    num_examples = []
    for i in range(len(synsets)):
        count = int(synsets[i][0][7][0][0])
        if count != 0:
            synset_ids.append(str(synsets[i][0][1][0]))
            num_examples.append(count)

    train_fpaths = []
    train_targets = []

    for i in range(len(synset_ids)):
        temp = glob.glob(ImageNet_fpath + 'Images/Train/' + synset_ids[i] + '/*.JPEG')
        train_fpaths += temp
        train_targets += len(temp)*[i]

    valid_fpaths = glob.glob(ImageNet_fpath + 'Images/Validation/*.JPEG')
    valid_file = open(ImageNet_fpath + 'DevKit/data/ILSVRC2012_validation_ground_truth.txt')
    valid_targets = []
    for _ in range(len(valid_fpaths)):
        valid_targets.append(int(valid_file.readline().replace('\n','')))

    train_fpaths, train_targets = shuffle(train_fpaths, train_targets)
    valid_fpaths, valid_targets = shuffle(valid_fpaths, valid_targets)

    return train_fpaths, train_targets, valid_fpaths, valid_targets

def generator(fpaths, targets):
    output = tf.read_file(fpaths[0])
    output = tf.image.decode_jpeg(contents=output, channels=3)
    output = tf.image.resize_image_with_crop_or_pad(output, IMAGE_SIZE, IMAGE_SIZE)
    output = tf.reshape(output, shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3])
    for i in range(len(fpaths)):
        if i >= 1:
            temp = tf.read_file(fpaths[i])
            temp = tf.image.decode_jpeg(contents=temp, channels=3)
            temp = tf.image.resize_image_with_crop_or_pad(temp, IMAGE_SIZE, IMAGE_SIZE)
            temp = tf.reshape(temp, shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3])
            output = tf.concat([output, temp], axis=0)

    return output.eval(), targets

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
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
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

def projection(convolved_input, input, decay, in_scope):
    convolved_shape = convolved_input.get_shape().as_list()
    shape = input.get_shape().as_list()
    with tf.variable_scope(in_scope) as scope:
        kernel = variable_with_weight_decay('weights',
                                            shape=[1,1, convolved_shape[3], shape[3]],
                                            stddev=5e-2,
                                            wd=decay)

        conv = tf.nn.conv2d(convolved_input, kernel, [1, 2, 2, 1], padding='SAME')
        biases = variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)

        output = tf.add(pre_activation, input)

    return output

def bottleneck_convolution(images, reducer, shape, decay, stride, in_scope):
    with tf.variable_scope(in_scope) as scope:
        embedding = convolution(images, [1,1,shape[2],reducer], decay, 1, 'reduce')

        conv = convolution(embedding, [shape[0],shape[1],reducer,reducer], decay, 1, 'conv')

        output = convolution(conv, [1,1,reducer,shape[3]], decay, stride, 'restore')

    return output
