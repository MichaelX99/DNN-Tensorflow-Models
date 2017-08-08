import scipy.io as sio
import glob
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import numpy as np

IMAGENET = '/home/mikep/hdd/DataSets/ImageNet2012/'
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
N_GPUS = 3
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_EPOCH = 10
NUM_CLASSES = 1000
DECAY = 0.0001

def read_imagenet(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class ImageNetRecord(object):
    pass
  result = ImageNetRecord()

  feature = {'valid/label': tf.FixedLenFeature([], tf.int64),
             'valid/height': tf.FixedLenFeature([], tf.int64),
             'valid/width': tf.FixedLenFeature([], tf.int64),
             'valid/channels': tf.FixedLenFeature([], tf.int64),
             'valid/image': tf.FixedLenFeature([], tf.string)}
  # Define a reader and read the next record
  reader = tf.TFRecordReader()
  result.key, serialized_example = reader.read(filename_queue)

  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature)

  # Convert the image data from string back to the numbers
  image = tf.decode_raw(features['valid/image'], tf.uint8)

  result.label = tf.cast(features['valid/label'], tf.int32)
  result.height = tf.cast(features['valid/height'], tf.int32)
  result.width = tf.cast(features['valid/width'], tf.int32)
  result.channels = tf.cast(features['valid/channels'], tf.int32)


  shape = tf.parallel_stack([result.height, result.width, result.channels])

  result.uint8image = tf.reshape(image, shape=shape)

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def generator(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'valid_%d.tfrecords' % i)
                 for i in range(128)]
    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_imagenet(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

    # Randomly flip the image horizontally.
    float_image = tf.image.random_flip_left_right(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'Validation/')
  images, labels = generator(data_dir=data_dir,
                             batch_size=FLAGS.batch_size)

  return images, labels


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
