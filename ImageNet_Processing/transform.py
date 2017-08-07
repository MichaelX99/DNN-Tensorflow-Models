import glob
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import scipy.io as sio
import cv2
import os

def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = np.shape(img)[0]
    width = np.shape(img)[1]

    mean, stddev = cv2.meanStdDev(img)
    adjusted_stddev = np.zeros(np.shape(stddev))
    adjusted_stddev[0] = np.max(stddev[0], 1.0/np.sqrt(img.size))
    adjusted_stddev[1] = np.max(stddev[1], 1.0/np.sqrt(img.size))
    adjusted_stddev[2] = np.max(stddev[2], 1.0/np.sqrt(img.size))

    img[:,:,0] = (img[:,:,0] - mean[0]) / adjusted_stddev[0]
    img[:,:,1] = (img[:,:,1] - mean[1]) / adjusted_stddev[1]
    img[:,:,2] = (img[:,:,2] - mean[2]) / adjusted_stddev[2]
    img = img.astype(np.uint8)

    return img, height, width


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

ImageNet_fpath = '/home/mikep/hdd/DataSets/ImageNet2012/'

meta = sio.loadmat(ImageNet_fpath + 'DevKit/data/meta.mat')
synsets = meta['synsets']

synset_ids = []
num_examples = []
for i in range(len(synsets)):
    count = int(synsets[i][0][7][0][0])
    if count != 0:
        synset_ids.append(str(synsets[i][0][1][0]))
        num_examples.append(count)

train_fpaths = []
train_targets = []

for i in range(len(synset_ids)):
    temp = glob.glob(ImageNet_fpath + 'Images/Train/' + synset_ids[i] + '/*.JPEG')
    train_fpaths += temp
    train_targets += len(temp)*[i]

valid_fpaths = glob.glob(ImageNet_fpath + 'Images/Validation/*.JPEG')
valid_file = open(ImageNet_fpath + 'DevKit/data/ILSVRC2012_validation_ground_truth.txt')
valid_targets = []
for _ in range(len(valid_fpaths)):
    valid_targets.append(int(valid_file.readline().replace('\n','')))

train_fpaths, train_targets = shuffle(train_fpaths, train_targets)
valid_fpaths, valid_targets = shuffle(valid_fpaths, valid_targets)

N_train = len(train_fpaths)
N_valid = len(valid_fpaths)

num_train = 32
num_valid = 8

train_per_file = N_train/num_train
valid_per_file = N_valid/num_valid

train_diff = N_train - (num_train*train_per_file)
valid_diff = N_valid - (num_valid*valid_per_file)
#############################################################################################
print("Starting Validation Transform")
if not os.path.isdir(ImageNet_fpath + 'TFRecord/Validation'):
    os.makedirs(ImageNet_fpath + 'TFRecord/Validation')
for i in range(num_valid):
    writer_fpath = ImageNet_fpath + 'TFRecord/Validation/valid_' + str(i) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(writer_fpath)

    if i != num_valid - 1:
        counter = valid_per_file + valid_diff
    else:
        counter = valid_per_file

    for j in range(counter):
        ind = i*valid_per_file + j
        if not j % 1000:
            print('percent done = ' +str(float(ind)/float(N_valid)) +'%')
        img, height, width = load_image(valid_fpaths[ind])
        label = valid_targets[ind]

        feature = {'valid/label': _int64_feature(label),
                   'valid/height': _int64_feature(height),
                   'valid/width': _int64_feature(width),
                   'valid/channels': _int64_feature(3),
                   'valid/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    print('Finished writing file #'+str(i))
#############################################################################################
print("Starting Training Transform")
if not os.path.isdir(ImageNet_fpath + 'TFRecord/Train'):
    os.makedirs(ImageNet_fpath + 'TFRecord/Train')
for i in range(num_train):
    writer_fpath = ImageNet_fpath + 'TFRecord/Train/train_' + str(i) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(writer_fpath)

    if i != num_train - 1:
        counter = train_per_file + train_diff
    else:
        counter = train_per_file

    for j in range(counter):
        ind = i*train_per_file + j
        if not j % 1000:
            print('percent done = ' +str(float(ind)/float(N_train)) +'%')
        img, height, width = load_image(train_fpaths[ind])
        label = train_targets[ind]

        feature = {'train/label': _int64_feature(label),
                   'train/height': _int64_feature(height),
                   'train/width': _int64_feature(width),
                   'train/channels': _int64_feature(3),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    print('Finished writing file #'+str(i))
