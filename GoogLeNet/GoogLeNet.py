import tensorflow as tf
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
import time

ImageNet_path = 'enter path'

train_fpath = glob.glob(ImageNet_path + 'train/*/*.png')
test_fpath = glob.glob(ImageNet_path + 'test/*/*.png')
num_classes = 1000

class layer_weight:
    def __init__(self, filter, input, output, mu, sigma, name, fc=False):
        with tf.name_scope(name):
            if fc == False:
                self.w = tf.Variable(tf.truncated_normal([filter, filter, input, output], mu, sigma))
                self.b = tf.Variable(tf.zeros(output))

            if fc == True:
                self.w = tf.Variable(tf.truncated_normal([input, output], mu, sigma))
                self.b = tf.Variable(tf.zeros(output))

            tf.summary.histogram('Weight', self.w)
            tf.add_to_collection('parameters', self.w)
            tf.summary.histogram('Bias', self.b)
            tf.add_to_collection('parameters', self.b)

class inception_weight:
    def __init__(self, input, one, red_three, three, red_five, five, pool_one, mu, sigma, name):
        with tf.name_scope(name):
            self.one = layer_weight(1, input, one, mu, sigma, 'one')

            self.red_three = layer_weight(1, input, red_three, mu, sigma, 'red_three')

            self.three = layer_weight(3, red_three, three, mu, sigma, 'three')

            self.red_five = layer_weight(1, input, red_five, mu, sigma, 'red_five')

            self.five = layer_weight(5, red_five, five, mu, sigma, 'five')

            self.pool = layer_weight(1, input, pool_one, mu, sigma, 'Pool')

class aux_weight:
    def __init__(self, input, map, mu, sigma, name):
        with tf.name_scope(name):
            self.red = layer_weight(1, input, 128, mu, sigma, 'Reduction')

            self.fc1 = layer_weight(None, map * map * 128, 1024, mu, sigma, 'FC1', fc=True)

            self.fc2 = layer_weight(None, 1024, 1000, mu, sigma, 'FC2', fc=True)

class weights:
    def __init__(self, mu, sigma):
        with tf.name_scope('Parameters'):
            self.l1 = layer_weight(7, 3, 64, mu, sigma, 'Layer1')
            self.l2 = layer_weight(3, 64, 192, mu, sigma, 'Layer2')
            self.inception1 = inception_weight(192, 64, 96, 128, 16, 32, 32, mu, sigma, 'Inception1')
            self.inception2 = inception_weight(256, 128, 128, 192, 32, 96, 64, mu, sigma, 'Inception2')
            self.inception3 = inception_weight(480, 192, 96, 208, 16, 48, 64, mu, sigma, 'Inception3')
            self.aux2 = aux_weight(512, 6, mu, sigma, 'Auxillary2')
            self.inception4 = inception_weight(512, 160, 112, 224, 24, 64, 64, mu, sigma, 'Inception4')
            self.inception5 = inception_weight(512, 128, 128, 256, 24, 64, 64, mu, sigma, 'Inception5')
            self.inception6 = inception_weight(512, 112, 144, 288, 32, 64, 64, mu, sigma, 'Inception6')
            self.aux1 = aux_weight(528, 6, mu, sigma, 'Auxillary1')
            self.inception7 = inception_weight(528, 256, 160, 320, 32, 128, 128, mu, sigma, 'Inception7')
            self.inception8 = inception_weight(832, 256, 160, 320, 32, 128, 128, mu, sigma, 'Inception8')
            self.inception9 = inception_weight(832, 384, 192, 384, 48, 128, 128, mu, sigma, 'Inception9')
            self.fc = layer_weight(None, 8*8*1024, 1000, mu, sigma, 'Fully_Connected', fc=True)

def convolve(x, w, b, stride, name):
    with tf.name_scope(name):
        out = tf.nn.conv2d(x, w, [1,stride,stride,1], 'SAME') + b
        out = tf.nn.relu(out)
        tf.summary.histogram(name, out)
    return out

def max_pool(x, size, stride, name):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'SAME')
        tf.summary.histogram(name, out)
    return out

def avg_pool(x, size, stride, name):
    with tf.name_scope(name):
        out = tf.nn.avg_pool(x, [1,size,size,1], [1,stride,stride,1], 'SAME')
        tf.summary.histogram(name, out)
    return out

def dropout(x, prob, name):
    with tf.name_scope(name):
        out = tf.nn.dropout(x, prob)
        tf.summary.histogram(name, out)
    return out

def fully_connected(x, w, b, name):
    with tf.name_scope(name):
        out = tf.matmul(x, w) + b
        tf.summary.histogram(name, out)
    return out

def inception(x, weights, name):
    with tf.name_scope(name):
        first = convolve(x, weights.one.w, weights.one.b, 1, '1x1')

        second = convolve(x, weights.red_three.w, weights.red_three.b, 1, 'reduce_3x3')
        second = tf.nn.relu(second)
        second = convolve(second, weights.three.w, weights.three.b, 1, '3x3')

        third = convolve(x, weights.red_five.w, weights.red_five.b, 1, 'reduce_5x5')
        third = tf.nn.relu(third)
        third = convolve(third, weights.five.w, weights.five.b, 1, '5x5')

        fourth = max_pool(x, 3, 1, 'Pool')
        fourth = convolve(fourth, weights.pool.w, weights.pool.b, 1, 'Pool')

        out = tf.concat([first, second, third, fourth], axis=3)
        out = tf.nn.relu(out)

        tf.summary.histogram(name, out)

    return out

def auxillary(x, weights, name):
    with tf.name_scope(name):
        pool = avg_pool(x, 5, 3, 'pool')

        reduced = convolve(pool, weights.red.w, weights.red.b, 1, 'red')
        reduced = tf.nn.relu(reduced)

        flat = tf.contrib.layers.flatten(reduced)

        fc1 = fully_connected(flat, weights.fc1.w, weights.fc1.b, 'fc1')
        fc1 = tf.nn.relu(fc1)

        drop = dropout(fc1, keep_prob2, 'dropout')

        fc2 = fully_connected(drop, weights.fc2.w, weights.fc2.b, 'fc2')
        fc2 = tf.nn.relu(fc2)

    return fc2

def network(x, weights):
    first = convolve(x, weights.l1.w, weights.l1.b, 2, 'Layer1')

    pool1 = max_pool(first, 3, 2, 'Pool1')

    second = convolve(pool1, weights.l2.w, weights.l2.b, 1, 'Layer2')

    pool2 = max_pool(second, 3, 2, 'Pool2')

    inception1 = inception(pool2, weights.inception1, 'Inception1')

    inception2 = inception(inception1, weights.inception2, 'Inception2')

    pool3 = max_pool(inception2, 3, 2, 'Pool3')

    inception3 = inception(pool3, weights.inception3, 'Inception3')

    aux2 = auxillary(inception3, weights.aux2, 'Auxillary2')

    inception4 = inception(inception3, weights.inception4, 'Inception4')

    inception5 = inception(inception4, weights.inception5, 'Inception5')

    inception6 = inception(inception5, weights.inception6, 'Inception6')

    aux1 = auxillary(inception6, weights.aux1, 'Auxillary1')

    inception7 = inception(inception6, weights.inception7, 'Inception7')

    pool4 = max_pool(inception7, 3, 2, 'Pool4')

    inception8 = inception(pool4, weights.inception8, 'Inception8')

    inception9 = inception(inception8, weights.inception9, 'Inception9')

    pool5 = avg_pool(inception9, 7, 1, 'Pool5')

    drop = dropout(pool5, keep_prob1, 'Dropout')

    flat = tf.contrib.layers.flatten(drop)

    fc = fully_connected(flat, weights.fc.w, weights.fc.b, 'Fully_Connected')

    return fc, aux1, aux2



tf_input = tf.placeholder(tf.float32, (None, 256, 256, 3))
tf_target = tf.placeholder(tf.int32, (None))
keep_prob1 = tf.placeholder(tf.float32, ())
keep_prob2 = tf.placeholder(tf.float32, ())


mu = 0
sigma = .01
weights = weights(mu, sigma)
logits1, logits2, logits3 = network(tf_input, weights)

one_hot_y = tf.one_hot(tf_target, num_classes)


with tf.name_scope('Cross_Entropy'):
    cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=one_hot_y)
    tf.summary.histogram('Cross_Entropy1', cross_entropy1)

    cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=one_hot_y)
    tf.summary.histogram('Cross_Entropy2', cross_entropy1)

    cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=one_hot_y)
    tf.summary.histogram('Cross_Entropy3', cross_entropy1)

with tf.name_scope('l2_loss'):
    l2_loss = [tf.nn.l2_loss(w) for w in tf.get_collection('parameters')]
    tf.summary.histogram('l2_loss', l2_loss)

with tf.name_scope('weight_magnitude'):
    weight_magnitude = tf.add_n(l2_loss)
    tf.summary.histogram('weight_magnitude', weight_magnitude)

with tf.name_scope('cross_entropy_loss'):
    cross_entropy_loss1 = tf.reduce_mean(cross_entropy1)
    tf.summary.scalar('Cross_entropy_loss1', cross_entropy_loss1)

    cross_entropy_loss2 = tf.multiply(tf.reduce_mean(cross_entropy2), 0.3)
    tf.summary.scalar('Cross_entropy_loss2', cross_entropy_loss2)

    cross_entropy_loss3 = tf.multiply(tf.reduce_mean(cross_entropy3), 0.3)
    tf.summary.scalar('Cross_entropy_loss3', cross_entropy_loss3)

with tf.name_scope('Loss'):
    loss_operation = cross_entropy_loss1 + cross_entropy_loss2 + cross_entropy_loss3 + l2_loss
    tf.summary.histogram('Loss', loss_operation)

# Define optimizer
global_step = 0
tf_epoch = tf.placeholder(tf.int32, 1)
starter_learning_rate = 0.00001
#learning_rate = tf.multiply(.96**tf_epoch, starter_learning_rate)
#tf.summary.scalar('learning_rate', learning_rate)

moment = .9
optimizer = tf.train.AdamOptimizer(starter_learning_rate)
#optimizer = tf.train.MomentumOptimizer(learning_rate, moment)

# Define the training operation
training_operation = optimizer.minimize(loss_operation)

# Define evaluation
with tf.name_scope('Evaluation'):
    correct_prediction = tf.equal(tf.argmax(logits1, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram('accuracy', accuracy_operation)

def data_gen(fpaths, labels):
    data = []
    for path in fpaths:
        temp = cv2.imread(path)
        temp = temp[:,:,::-1]
        temp = cv2.resize(temp, (size, size))
        data.append(temp)
    return data, labels

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = data_gen(X_data[offset:end], y_data[offset:end])
        accuracy = sess.run(accuracy_operation, feed_dict={tf_input: batch_x, tf_target: batch_y, keep_prob1: 1.0, keep_prob2: 0.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples#, out

tensor_summary = tf.summary.merge_all()

# Initialize lists for plotting
validation = []
training = []
t = []


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('train',sess.graph)

    print("Training")
    start_time = time.clock()
    for i in range(5):
        for offset in range(0, len(train_fpaths), batch_size):
        #for offset in range(0, batch_size, batch_size):
            end = offset + batch_size
            batch_x, batch_y = data_gen(train_fpaths[offset:end], y_train[offset:end])
            _, out_summary = sess.run([training_operation, tensor_summary], feed_dict={tf_input: batch_x, tf_target: batch_y, keep_prob1: .5, keep_prob2: .7})
            writer.add_summary(out_summary,i)

        # Evaluate the network
        validation_accuracy = evaluate(test_fpaths, y_test, batch_size)
        training_accuracy = evaluate(train_fpaths, y_train, batch_size)

        # Append values to graph later
        validation.append(validation_accuracy)
        training.append(training_accuracy)
        t.append(i)


        # Print training status
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print()

    #saver.save(sess, './alexnet/alexnet')
    writer.close()
    print("Model saved")
    print('seconds elapsed = ',time.clock()-start_time)
"""
# Visualize the learning
plt.plot(t, validation,t,training)
plt.legend(('Validation','Training'))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning')
"""
