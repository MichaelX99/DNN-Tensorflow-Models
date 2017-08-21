import tensorflow as tf
import helper

def inference(images):
    k = 32
    b1 = 6
    b2 = 12
    b3 = 24
    b4 = 16

    # conv1
    conv1 = helper.convolution(images, [7,7,3,k], helper.DECAY, 2, 'conv1')

    # pool1
    pool1 = tf.nn.max_pool(conv1, [1,3,3,1], [1,2,2,1], 'SAME', name='pool1')

    input1 = helper.dense_block(pool1, k, b1, 'conv1')
    transition1 = helper.transition(input1, k, b1, 'transition1')

    input2 = helper.dense_block(transition1, k, b2, 'conv2')
    transition2 = helper.transition(input2, k, b2, 'transition2')

    input3 = helper.dense_block(transition2, k, b3, 'conv3')
    transition3 = helper.transition(input3, k, b3, 'transition3')

    input4 = helper.dense_block(transition3, k, b4, 'conv4')

    avg_pool = tf.nn.avg_pool(input4, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
    dense = tf.contrib.layers.flatten(avg_pool)
    with tf.variable_scope("fc") as scope:
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=[51200,helper.NUM_CLASSES],
                                            stddev=5e-2,
                                            wd=helper.DECAY)

        conv = tf.matmul(dense, kernel)
        biases = helper.variable_on_cpu('biases', helper.NUM_CLASSES, tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)

    return output
