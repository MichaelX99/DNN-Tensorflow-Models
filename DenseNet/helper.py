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
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_EPOCH = 10
NUM_CLASSES = 1000
DECAY = 0.0001

def read_imagenet(serialized_example):
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

  feature = {'train/label': tf.FixedLenFeature([], tf.int64),
             'train/height': tf.FixedLenFeature([], tf.int64),
             'train/width': tf.FixedLenFeature([], tf.int64),
             'train/channels': tf.FixedLenFeature([], tf.int64),
             'train/image': tf.FixedLenFeature([], tf.string)}

  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature)

  # Convert the image data from string back to the numbers
  image = tf.decode_raw(features['train/image'], tf.uint8)

  label = tf.cast(features['train/label'], tf.int32)
  height = tf.cast(features['train/height'], tf.int32)
  width = tf.cast(features['train/width'], tf.int32)
  channels = tf.cast(features['train/channels'], tf.int32)


  shape = tf.parallel_stack([height, width, channels])

  image = tf.reshape(image, shape=shape)
  image = tf.cast(image, tf.float32)

  return image, label

def image_preprocessing(image, thread_id):
  """Decode and preprocess one image for evaluation or training.

  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  #image = distort_image(image, height, width, thread_id)

  image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  image.set_shape([IMAGE_SIZE,IMAGE_SIZE,3])

  return image

def generator():
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    data_dir = os.path.join(IMAGENET, 'TFRecord/Train/')
    data_files = [os.path.join(data_dir, 'train_%d.tfrecords' % i)
                 for i in range(1024)]

    for f in data_files:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('batch_processing'):
        # Create filename_queue
        filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)

        num_preprocess_threads = 4
        num_readers = 4
        input_queue_memory_factor = 16

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * input_queue_memory_factor

        examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * BATCH_SIZE,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))

        example_serialized = examples_queue.dequeue()

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
          # Parse a serialized Example proto to extract the image and metadata.
          image, label = read_imagenet(example_serialized)

          image = image_preprocessing(image, thread_id)

          images_and_labels.append([image, label])

        images, label_index_batch = tf.train.batch_join(
          images_and_labels,
          batch_size=BATCH_SIZE,
          capacity=2 * num_preprocess_threads * BATCH_SIZE)

        # Reshape images into these desired dimensions.
        depth = 3
        images = tf.reshape(images, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, depth])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

    return images, tf.reshape(label_index_batch, [BATCH_SIZE])


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
