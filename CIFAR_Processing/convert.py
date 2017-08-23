#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:41:13 2017

@author: mikep
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import matplotlib

num_files = 5
IMAGE_SIZE = 32
fpath = '/home/mikep/DataSets/CIFAR10/'
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
int_classes = [0,1,2,3,4,5,6,7,8,9]

both_classes = []
for i in range(len(classes)):
    temp_class = classes[i]
    temp_int = int_classes[i]
    both_classes.append((temp_class, temp_int))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def split_images(data):
    images = []
    N = len(data)
    size = IMAGE_SIZE * IMAGE_SIZE
    for i in range(N):
        temp_image = data[i]
        temp_R = temp_image[:size]
        temp_G = temp_image[size:2*size]
        temp_B = temp_image[2*size:3*size]
        temp_R = np.reshape(temp_R, (IMAGE_SIZE,IMAGE_SIZE))
        temp_G = np.reshape(temp_G, (IMAGE_SIZE,IMAGE_SIZE))
        temp_B = np.reshape(temp_B, (IMAGE_SIZE,IMAGE_SIZE))
        temp_image = np.dstack((temp_R, temp_G, temp_B))
        images.append(temp_image)
        
    return images

def split_dict(dictionary):
    data = dictionary[b'data']
    temp_filenames = dictionary[b'filenames']
    labels = dictionary[b'labels']
    
    filenames = []
    for path in temp_filenames:
        filenames.append(path.decode("utf-8"))
    
    images = split_images(data)
    
    return images, filenames, labels



temp_dict = unpickle(fpath+'test_batch')
test_images, test_filenames, test_labels = split_dict(temp_dict)

images = []
labels = []
filenames = []
for i in range(num_files):
    temp_dict = unpickle(fpath+'data_batch_'+str(i+1))
    temp_images, temp_filenames, temp_labels = split_dict(temp_dict)
    images.append(temp_images)
    filenames.append(temp_filenames)
    labels.append(temp_labels)
    
images = list(itertools.chain.from_iterable(images))
filenames = list(itertools.chain.from_iterable(filenames))
labels = list(itertools.chain.from_iterable(labels))
    
if not os.path.exists(fpath+'Train'):
    os.makedirs(fpath+'Train')

if not os.path.exists(fpath+'Test'):
    os.makedirs(fpath+'Test')
    
label_file = open(fpath + 'labels.txt', 'w')
for name in both_classes:
    label_file.write(name[0]+'\n')
    
    if not os.path.exists(fpath+'Train/'+name[0]):
        os.makedirs(fpath+'Train/'+name[0])
    
    if not os.path.exists(fpath+'Test/'+name[0]):
        os.makedirs(fpath+'Test/'+name[0])
        
    for i in range(len(images)):
        if labels[i] == name[1]:
            matplotlib.image.imsave(fpath + 'Train/' + name[0] + '/' + filenames[i], images[i])
            
    for i in range(len(test_images)):
        if test_labels[i] == name[1]:
            matplotlib.image.imsave(fpath + 'Test/' + name[0] + '/' + test_filenames[i], test_images[i])
            
label_file.close()