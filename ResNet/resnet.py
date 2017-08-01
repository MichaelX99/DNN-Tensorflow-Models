import tensorflow as tf
import helper

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
N_GPUS = 2
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_EPOCH = 10
NUM_CLASSES = 1000
DECAY = 0.0001

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
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=shape,
                                            stddev=5e-2,
                                            wd=decay)

        conv = tf.nn.conv2d(images, kernel, [1, stride, stride, 1], padding='SAME')
        biases = helper.variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        batch_norm = tf.contrib.layers.batch_norm(conv)

    return batch_norm

def projection(convolved_input, input, decay, in_scope):
    convolved_shape = convolved_input.get_shape().as_list()
    shape = input.get_shape().as_list()
    with tf.variable_scope(in_scope) as scope:
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=[1,1, convolved_shape[3], shape[3]],
                                            stddev=5e-2,
                                            wd=decay)

        conv = tf.nn.conv2d(convolved_input, kernel, [1, 2, 2, 1], padding='SAME')
        biases = helper.variable_on_cpu('biases', shape[3], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)

    return pre_activation

def debug_inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = helper.variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = helper.variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)


  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = helper.variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = helper.variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
    dim = reshape.get_shape()[1].value
    weights = helper.variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = helper.variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = helper.variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = helper.variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = helper.variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = helper.variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    print(softmax_linear.get_shape().as_list())

  return softmax_linear


def inference(images):
    # conv1
    conv1 = convolution(images, [7, 7, 3, 64], DECAY, 2, 'conv1')


    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                        padding='SAME', name='pool1')

    # conv2
    conv2 = convolution(pool1, [3,3,64,64], DECAY, 1, 'conv2')

    # conv3
    conv3 = convolution(conv2, [3,3,64,64], DECAY, 1, 'conv3') + pool1

    # conv4
    conv4 = convolution(conv3, [3,3,64,64], DECAY, 1, 'conv4')

    # conv5
    conv5 = convolution(conv4, [3,3,64,64], DECAY, 1, 'conv5') + conv3

    # conv6
    conv6 = convolution(conv5, [3,3,64,64], DECAY, 1, 'conv6')

    # conv7
    conv7 = convolution(conv6, [3,3,64,64], DECAY, 1, 'conv7') + conv5

    # conv8
    conv8 = convolution(conv7, [3,3,64,128], DECAY, 2, 'conv8')

    # conv9
    conv9 = convolution(conv8, [3,3,128,128], DECAY, 1, 'conv9')

    # projection1
    proj1 = projection(conv7, conv9, DECAY, 'proj1')

    # conv10
    conv10 = convolution(proj1, [3,3,128,128], DECAY, 1, 'conv10')

    # conv11
    conv11 = convolution(conv10, [3,3,128,128], DECAY, 1, 'conv11') + proj1

    # conv12
    conv12 = convolution(conv11, [3,3,128,128], DECAY, 1, 'conv12')

    # conv13
    conv13 = convolution(conv12, [3,3,128,128], DECAY, 1, 'conv13') + conv11

    # conv14
    conv14 = convolution(conv13, [3,3,128,128], DECAY, 1, 'conv14')

    # conv15
    conv15 = convolution(conv14, [3,3,128,128], DECAY, 1, 'conv15') + conv13

    # conv16
    conv16 = convolution(conv15, [3,3,128,256], DECAY, 2, 'conv16')

    # conv17
    conv17 = convolution(conv16, [3,3,256,256], DECAY, 1, 'conv17')

    # projection2
    proj2 = projection(conv15, conv17, DECAY, 'proj2')

    # conv18
    conv18 = convolution(proj2, [3,3,256,256], DECAY, 1, 'conv18')

    # conv19
    conv19 = convolution(conv18, [3,3,256,256], DECAY, 1, 'conv19') + proj2

    # conv20
    conv20 = convolution(conv19, [3,3,256,256], DECAY, 1, 'conv20')

    # conv21
    conv21 = convolution(conv20, [3,3,256,256], DECAY, 1, 'conv21') + conv19

    # conv22
    conv22 = convolution(conv21, [3,3,256,256], DECAY, 1, 'conv22')

    # conv23
    conv23 = convolution(conv22, [3,3,256,256], DECAY, 1, 'conv23') + conv21

    # conv24
    conv24 = convolution(conv23, [3,3,256,256], DECAY, 1, 'conv24')

    # conv25
    conv25 = convolution(conv24, [3,3,256,256], DECAY, 1, 'conv25') + conv23

    # conv26
    conv26 = convolution(conv25, [3,3,256,256], DECAY, 1, 'conv26')

    # conv27
    conv27 = convolution(conv26, [3,3,256,256], DECAY, 1, 'conv27') + conv25

    # conv28
    conv28 = convolution(conv27, [3,3,256,512], DECAY, 2, 'conv28')

    # conv29
    conv29 = convolution(conv28, [3,3,512,512], DECAY, 1, 'conv29')

    # projection3
    proj3 = projection(conv27, conv29, DECAY, 'proj3')

    # conv30
    conv30 = convolution(proj3, [3,3,512,512], DECAY, 1, 'conv30')

    # conv31
    conv31 = convolution(conv30, [3,3,512,512], DECAY, 1, 'conv31') + proj3

    # conv32
    conv32 = convolution(conv31, [3,3,512,512], DECAY, 1, 'conv32')

    # conv33
    conv33 = convolution(conv32, [3,3,512,512], DECAY, 1, 'conv33') + conv31

    # average pool
    conv_shape = conv33.get_shape().as_list()
    avg_pool = tf.nn.avg_pool(conv33, ksize=[1,conv_shape[1],conv_shape[1],1], strides=[1,1,1,1], padding='SAME')
    dense = tf.contrib.layers.flatten(avg_pool)
    dense_shape = dense.get_shape().as_list()

    with tf.variable_scope("fc") as scope:
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=[dense_shape[1],NUM_CLASSES],
                                            stddev=5e-2,
                                            wd=DECAY)

        conv = tf.matmul(dense, kernel)
        biases = helper.variable_on_cpu('biases', NUM_CLASSES, tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)

    return output
