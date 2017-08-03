import tensorflow as tf
import helper

def inference(images):
    # conv1
    conv1 = helper.convolution(images, [7, 7, 3, 64], helper.DECAY, 2, 'conv1')


    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                        padding='SAME', name='pool1')

    # conv2
    conv2 = helper.convolution(pool1, [3,3,64,64], helper.DECAY, 1, 'conv2')

    # conv3
    conv3 = helper.convolution(conv2, [3,3,64,64], helper.DECAY, 1, 'conv3') + pool1

    # conv4
    conv4 = helper.convolution(conv3, [3,3,64,64], helper.DECAY, 1, 'conv4')

    # conv5
    conv5 = helper.convolution(conv4, [3,3,64,64], helper.DECAY, 1, 'conv5') + conv3

    # conv6
    conv6 = helper.convolution(conv5, [3,3,64,64], helper.DECAY, 1, 'conv6')

    # conv7
    conv7 = helper.convolution(conv6, [3,3,64,64], helper.DECAY, 1, 'conv7') + conv5

    # conv8
    conv8 = helper.convolution(conv7, [3,3,64,128], helper.DECAY, 2, 'conv8')

    # conv9
    conv9 = helper.convolution(conv8, [3,3,128,128], helper.DECAY, 1, 'conv9')

    # projection1
    proj1 = helper.projection(conv7, conv9, helper.DECAY, 'proj1')

    # conv10
    conv10 = helper.convolution(proj1, [3,3,128,128], helper.DECAY, 1, 'conv10')

    # conv11
    conv11 = helper.convolution(conv10, [3,3,128,128], helper.DECAY, 1, 'conv11') + proj1

    # conv12
    conv12 = helper.convolution(conv11, [3,3,128,128], helper.DECAY, 1, 'conv12')

    # conv13
    conv13 = helper.convolution(conv12, [3,3,128,128], helper.DECAY, 1, 'conv13') + conv11

    # conv14
    conv14 = helper.convolution(conv13, [3,3,128,128], helper.DECAY, 1, 'conv14')

    # conv15
    conv15 = helper.convolution(conv14, [3,3,128,128], helper.DECAY, 1, 'conv15') + conv13

    # conv16
    conv16 = helper.convolution(conv15, [3,3,128,256], helper.DECAY, 2, 'conv16')

    # conv17
    conv17 = helper.convolution(conv16, [3,3,256,256], helper.DECAY, 1, 'conv17')

    # projection2
    proj2 = helper.projection(conv15, conv17, helper.DECAY, 'proj2')

    # conv18
    conv18 = helper.convolution(proj2, [3,3,256,256], helper.DECAY, 1, 'conv18')

    # conv19
    conv19 = helper.convolution(conv18, [3,3,256,256], helper.DECAY, 1, 'conv19') + proj2

    # conv20
    conv20 = helper.convolution(conv19, [3,3,256,256], helper.DECAY, 1, 'conv20')

    # conv21
    conv21 = helper.convolution(conv20, [3,3,256,256], helper.DECAY, 1, 'conv21') + conv19

    # conv22
    conv22 = helper.convolution(conv21, [3,3,256,256], helper.DECAY, 1, 'conv22')

    # conv23
    conv23 = helper.convolution(conv22, [3,3,256,256], helper.DECAY, 1, 'conv23') + conv21

    # conv24
    conv24 = helper.convolution(conv23, [3,3,256,256], helper.DECAY, 1, 'conv24')

    # conv25
    conv25 = helper.convolution(conv24, [3,3,256,256], helper.DECAY, 1, 'conv25') + conv23

    # conv26
    conv26 = helper.convolution(conv25, [3,3,256,256], helper.DECAY, 1, 'conv26')

    # conv27
    conv27 = helper.convolution(conv26, [3,3,256,256], helper.DECAY, 1, 'conv27') + conv25

    # conv28
    conv28 = helper.convolution(conv27, [3,3,256,512], helper.DECAY, 2, 'conv28')

    # conv29
    conv29 = helper.convolution(conv28, [3,3,512,512], helper.DECAY, 1, 'conv29')

    # projection3
    proj3 = helper.projection(conv27, conv29, helper.DECAY, 'proj3')

    # conv30
    conv30 = helper.convolution(proj3, [3,3,512,512], helper.DECAY, 1, 'conv30')

    # conv31
    conv31 = helper.convolution(conv30, [3,3,512,512], helper.DECAY, 1, 'conv31') + proj3

    # conv32
    conv32 = helper.convolution(conv31, [3,3,512,512], helper.DECAY, 1, 'conv32')

    # conv33
    conv33 = helper.convolution(conv32, [3,3,512,512], helper.DECAY, 1, 'conv33') + conv31

    # average pool
    conv_shape = conv33.get_shape().as_list()
    avg_pool = tf.nn.avg_pool(conv33, ksize=[1,conv_shape[1],conv_shape[1],1], strides=[1,1,1,1], padding='SAME')
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
