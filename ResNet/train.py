from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import time
import resnet34_model
import resnet50_model
import helper

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
  #logits = resnet34_model.inference(images)
  logits = resnet50_model.inference(images)

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
        train_fpaths, train_targets, valid_fpaths, valid_targets = helper.import_ImageNet(helper.IMAGENET)

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Learning rate
        lr = .001

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)

        input_tensor = tf.placeholder(tf.float32, (helper.BATCH_SIZE, helper.IMAGE_SIZE, helper.IMAGE_SIZE, 3))
        input_target = tf.placeholder(tf.int32, (helper.BATCH_SIZE))
        one_hot_target = tf.one_hot(input_target, helper.NUM_CLASSES)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(helper.N_GPUS+1):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (helper.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, input_tensor, one_hot_target)

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
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(init)

            #for epoch in range(helper.MAX_EPOCH):
            for epoch in range(1):
                train_loss = 0
                valid_loss = 0
                times_run = 0
                for offset in range(0, len(train_fpaths), helper.BATCH_SIZE):
                #for offset in range(0, helper.BATCH_SIZE, helper.BATCH_SIZE):
                    end = offset + helper.BATCH_SIZE
                    batch_x, batch_y = helper.generator(train_fpaths[offset:end], train_targets[offset:end])
                    _, loss_value = sess.run([train_op, loss], feed_dict={input_tensor: batch_x, input_target: batch_y})
                    train_loss += loss_value

                    batch_x, batch_y = helper.generator(valid_fpaths[offset:end], valid_targets[offset:end])
                    loss_value = sess.run(loss, feed_dict={input_tensor: batch_x, input_target: batch_y})
                    valid_loss += loss_value

                    tower_grads[:] = []
                    times_run += 1

                train_loss /= times_run
                valid_loss /= times_run
                print("epoch: "+str(epoch)+", training loss = "+str(train_loss)+", validation loss = "+str(valid_loss))

if __name__ == '__main__':
    train()
