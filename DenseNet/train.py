from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import time
from datetime import datetime
import densenet
import helper

from imagenet_data import ImagenetData
import image_processing

def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = densenet.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = helper.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Learning rate
        lr = .001

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        split_batch_size = int(helper.BATCH_SIZE / helper.N_GPUS)
        num_preprocess_threads = helper.NUM_THREADS * helper.N_GPUS

        # Get images and labels for CIFAR-10.
        dataset = ImagenetData(subset='train')
        assert dataset.data_files()

        assert helper.BATCH_SIZE % helper.N_GPUS == 0, ('Batch size must be divisible by number of GPUs')
        split_batch_size = int(helper.BATCH_SIZE / helper.N_GPUS)

        # Override the number of preprocessing threads to account for the increased
        # number of GPU towers.
        num_preprocess_threads = helper.NUM_THREADS * helper.N_GPUS
        images, labels = image_processing.distorted_inputs(dataset, batch_size=helper.BATCH_SIZE, num_preprocess_threads=num_preprocess_threads)

        # Split the batch of images and labels for towers.
        images_splits = tf.split(axis=0, num_or_size_splits=helper.N_GPUS, value=images)
        labels_splits = tf.split(axis=0, num_or_size_splits=helper.N_GPUS, value=labels)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(helper.N_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (helper.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, images_splits[i], labels_splits[i])

                        tf.get_variable_scope().reuse_variables()

                        grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)


        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(helper.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        print("training")

        #for epoch in range(helper.MAX_EPOCH):
        for epoch in range(helper.MAX_STEPS):

            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            num_examples_per_step = helper.BATCH_SIZE * helper.N_GPUS
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / helper.N_GPUS

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))

if __name__ == '__main__':
    train()
