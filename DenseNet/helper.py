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
NUM_CLASSES = 1000 + 1
DECAY = 0.0001
TRAIN_SHARDS = 128
VALIDATION_SHARDS = 24
NUM_THREADS = 8

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

  feature = {'image/class/label': tf.FixedLenFeature([], tf.int64),
             'image/height': tf.FixedLenFeature([], tf.int64),
             'image/width': tf.FixedLenFeature([], tf.int64),
             'image/channels': tf.FixedLenFeature([], tf.int64),
             'image/encoded': tf.FixedLenFeature([], tf.string)}

  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature)

  label = tf.cast(features['image/class/label'], tf.int32)
  height = tf.cast(features['image/height'], tf.int32)
  width = tf.cast(features['image/width'], tf.int32)
  channels = tf.cast(features['image/channels'], tf.int32)

  return features['image/encoded'], label, height, width, channels

 def distort_color(image, thread_id=0, scope=None):
  """Distort the color of the image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for name_scope.
  Returns:
    color-distorted image
  """
  with tf.name_scope(values=[image], name=scope, default_name='distort_color'):
    color_ordering = thread_id % 2

    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
return image

def image_preprocessing(image_buffer, height, width, channels, thread_id):
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
  image = tf.image.decode_jpeg(image_buffer, channels=channels)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  distorted_image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)

  distorted_image.set_shape([IMAGE_SIZE,IMAGE_SIZE,channels])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Randomly distort the colors.
  distorted_image = distort_color(distorted_image, thread_id)

  # Finally, rescale to [-1,1] instead of [0, 1)
  distorted_image = tf.subtract(image, 0.5)
  distorted_image = tf.multiply(image, 2.0)

  return distorted_image

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
    data_files = [os.path.join(data_dir, 'train_%d.tfrecords' % i)for i in range(TRAIN_SHARDS)]

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
          image_buffer, label, height, width, channels = read_imagenet(example_serialized)

          image = image_preprocessing(image_buffer, height, width, channels, thread_id)

          images_and_labels.append([image, label])

        images, label_index_batch = tf.train.batch_join(
          images_and_labels,
          batch_size=BATCH_SIZE,
          capacity=2 * num_preprocess_threads * BATCH_SIZE)

        # Reshape images into these desired dimensions.
        images = tf.cast(images, tf.float32)
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

def transition(input, k, scope):
    transition = helper.convolution(input, [1,1,k,k], helper.DECAY, 1, scope)
    transition = tf.nn.max_pool(transition, [1,3,3,1], [1,2,2,1], 'SAME')

    return transition

def dense_block(input, k, b, scope):
    conv = helper.convolution(input, [1,1,k,k], helper.DECAY, 1, scope+'_1')
    conv = helper.convolution(conv, [3,3,k,k], helper.DECAY, 1, scope+'_2')
    for i in range(b - 1):
        l = helper.convolution(conv, [1,1,k,k], helper.DECAY, 1, scope + '_' + str(i+2) + '_1')

        l = helper.convolution(l, [3,3,k,k], helper.DECAY, 1, scope + '_' + str(i+2) + '_2')

        conv = tf.concat([conv, l], axis=3)

    return conv
