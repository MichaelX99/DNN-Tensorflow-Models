import scipy.io as sio
import glob
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
import resnet

IMAGENET = '/home/mikep/hdd/DataSets/ImageNet2012/'


def import_ImageNet(ImageNet_fpath):
    """Helper to import ImageNet

    Args:
      ImageNet_fpath: path to ImageNet

    Returns:
      train_fpaths: list of filepaths to every training example
      train_target: list of numeric class labels for the training examples
      valid_fpaths: list of filepaths to every validation example
      valid_target: list of numeric class labels for the validation examples
    """
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

    return train_fpaths, train_targets, valid_fpaths, valid_targets

def generator(fpaths, targets):
    images = []
    for path in fpaths:
        temp = cv2.imread(path)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = tf.image.resize_image_with_crop_or_pad(temp, resnet.IMAGE_SIZE, resnet.IMAGE_SIZE)
        images.append(temp)

    return images, targets

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
