import tensorflow as tf
import helper

def inference(input):
    conv1 = helper.convolution(input, [7, 7, 3, 64], helper.DECAY, 2, 'conv1')

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    block1_1 = helper.block_convolution(pool1, 64, helper.DECAY, 1, 'block1_1') + pool1

    block1_2 = helper.block_convolution(block1_1, 64, helper.DECAY, 1, 'block1_2') + block1_1

    block1_3 = helper.block_convolution(block1_2, 64, helper.DECAY, 2, 'block1_3') + helper.convolution(block1_2, [1,1,64,64], helper.DECAY, 2, 'proj1')

    block2_1 = helper.block_convolution(block1_3, 128, helper.DECAY, 1, 'block2_1') + helper.convolution(block1_3, [1,1,64,128], helper.DECAY, 1, 'proj2')

    block2_2 = helper.block_convolution(block2_1, 128, helper.DECAY, 1, 'block2_2') + block2_1

    block2_3 = helper.block_convolution(block2_2, 128, helper.DECAY, 1, 'block2_3') + block2_2

    block2_4 = helper.block_convolution(block2_3, 128, helper.DECAY, 2, 'block2_4') + helper.convolution(block2_3, [1,1,128,128], helper.DECAY, 2, 'proj3')

    block3_1 = helper.block_convolution(block2_4, 256, helper.DECAY, 1, 'block3_1') + helper.convolution(block2_4, [1,1,128,256], helper.DECAY, 1, 'proj4')

    block3_2 = helper.block_convolution(block3_1, 256, helper.DECAY, 1, 'block3_2') + block3_1

    block3_3 = helper.block_convolution(block3_2, 256, helper.DECAY, 1, 'block3_3') + block3_2

    block3_4 = helper.block_convolution(block3_3, 256, helper.DECAY, 1, 'block3_4') + block3_3

    block3_5 = helper.block_convolution(block3_4, 256, helper.DECAY, 1, 'block3_5') + block3_4

    block3_6 = helper.block_convolution(block3_5, 256, helper.DECAY, 2, 'block3_6') + helper.convolution(block3_5, [1,1,256,256], helper.DECAY, 2, 'proj5')

    block4_1 = helper.block_convolution(block3_6, 512, helper.DECAY, 1, 'block4_1') + helper.convolution(block3_6, [1,1,256,512], helper.DECAY, 1, 'proj6')

    block4_2 = helper.block_convolution(block4_1, 512, helper.DECAY, 1, 'block4_2') + block4_1

    block4_3 = helper.block_convolution(block4_2, 512, helper.DECAY, 1, 'block4_3') + block4_2

    avg_pool = tf.nn.avg_pool(block4_3, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
    dense = tf.contrib.layers.flatten(avg_pool)

    with tf.variable_scope("fc") as scope:
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=[25088,helper.NUM_CLASSES],
                                            stddev=5e-2,
                                            wd=helper.DECAY)

        conv = tf.matmul(dense, kernel)
        biases = helper.variable_on_cpu('biases', helper.NUM_CLASSES, tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)

    return output
