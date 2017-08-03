import tensorflow as tf
import helper

def inference(images):
    # conv1
    conv1 = helper.convolution(images, [7, 7, 3, 64], helper.DECAY, 2, 'conv1')


    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                        padding='SAME', name='pool1')

    rescale1 = helper.convolution(pool1, [1, 1, 64, 256], helper.DECAY, 1, 'rescale1')

    # conv2_B
    conv2_B = helper.bottleneck_convolution(pool1, 64, [3,3,64,256], helper.DECAY, 1, 'conv2_B') + rescale1

    # conv3_B
    conv3_B = helper.bottleneck_convolution(conv2_B, 64, [3,3,256,256], helper.DECAY, 1, 'conv3_B') + conv2_B

    # conv4_B
    conv4_B = helper.bottleneck_convolution(conv3_B, 64, [3,3,256,256], helper.DECAY, 2, 'conv4_B') + helper.convolution(conv3_B, [1, 1, 256, 256], helper.DECAY, 2, 'block1')

    # projection1
    proj1 = helper.projection(conv3_B, conv4_B, helper.DECAY, 'proj1')

    rescale2 = helper.convolution(proj1, [1, 1, 256, 512], helper.DECAY, 1, 'rescale2')

    # conv5_B
    conv5_B = helper.bottleneck_convolution(proj1, 128, [3,3,256,512], helper.DECAY, 1, 'conv5_B') + rescale2

    # conv6_B
    conv6_B = helper.bottleneck_convolution(conv5_B, 128, [3,3,512,512], helper.DECAY, 1, 'conv6_B') + conv5_B

    # conv7_B
    conv7_B = helper.bottleneck_convolution(conv6_B, 128, [3,3,512,512], helper.DECAY, 1, 'conv7_B') + conv6_B

    # conv8_B
    conv8_B = helper.bottleneck_convolution(conv7_B, 128, [3,3,512,512], helper.DECAY, 2, 'conv8_B') + helper.convolution(conv7_B, [1, 1, 512, 512], helper.DECAY, 2, 'block2')

    # projection2
    proj2 = helper.projection(conv7_B, conv8_B, helper.DECAY, 'proj2')

    rescale3 = helper.convolution(proj2, [1, 1, 512, 1024], helper.DECAY, 1, 'rescale3')

    # conv9_B
    conv9_B = helper.bottleneck_convolution(proj2, 256, [3,3,512,1024], helper.DECAY, 1, 'conv9_B') + rescale3

    # conv10_B
    conv10_B = helper.bottleneck_convolution(conv9_B, 256, [3,3,1024,1024], helper.DECAY, 1, 'conv10_B') + conv9_B

    # conv11_B
    conv11_B = helper.bottleneck_convolution(conv10_B, 256, [3,3,1024,1024], helper.DECAY, 1, 'conv11_B') + conv10_B

    # conv12_B
    conv12_B = helper.bottleneck_convolution(conv11_B, 256, [3,3,1024,1024], helper.DECAY, 1, 'conv12_B') + conv11_B

    # conv13_B
    conv13_B = helper.bottleneck_convolution(conv12_B, 256, [3,3,1024,1024], helper.DECAY, 1, 'conv13_B') + conv12_B

    # conv14_B
    conv14_B = helper.bottleneck_convolution(conv13_B, 256, [3,3,1024,1024], helper.DECAY, 2, 'conv14_B') + helper.convolution(conv13_B, [1, 1, 1024, 1024], helper.DECAY, 2, 'block3')

    # projection3
    proj3 = helper.projection(conv13_B, conv14_B, helper.DECAY, 'proj3')

    rescale4 = helper.convolution(proj3, [1, 1, 1024, 2048], helper.DECAY, 1, 'rescale4')

    # conv15_B
    conv15_B = helper.bottleneck_convolution(proj3, 512, [3,3,1024,2048], helper.DECAY, 1, 'conv15_B') + rescale4

    # conv16_B
    conv16_B = helper.bottleneck_convolution(conv15_B, 512, [3,3,2048,2048], helper.DECAY, 1, 'conv16_B') + conv15_B

    # conv17_B
    conv17_B = helper.bottleneck_convolution(conv16_B, 512, [3,3,2048,2048], helper.DECAY, 2, 'conv17_B') + helper.convolution(conv16_B, [1, 1, 2048, 2048], helper.DECAY, 2, 'block4')


    # projection4
    proj4 = helper.projection(conv16_B, conv17_B, helper.DECAY, 'proj4')

    # average pool
    conv_shape = proj4.get_shape().as_list()
    avg_pool = tf.nn.avg_pool(proj4, ksize=[1,conv_shape[1],conv_shape[1],1], strides=[1,1,1,1], padding='SAME')
    dense = tf.contrib.layers.flatten(avg_pool)
    dense_shape = dense.get_shape().as_list()

    with tf.variable_scope("fc") as scope:
        kernel = helper.variable_with_weight_decay('weights',
                                            shape=[dense_shape[1],helper.NUM_CLASSES],
                                            stddev=5e-2,
                                            wd=helper.DECAY)

        conv = tf.matmul(dense, kernel)
        biases = helper.variable_on_cpu('biases', helper.NUM_CLASSES, tf.constant_initializer(0.0))
        output = tf.nn.bias_add(conv, biases)

    return output
