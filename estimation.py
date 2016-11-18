#!/usr/bin/python
print 'Disparity Map Estimation';

import cv2
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import os
if os.path.isfile('model/liberty.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_cpu()

model_def = 'model/notredame_deploy.txt'
model_weights = 'model/notredame.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (1,0))  # swap channels from RGB to BGR

# We need to reshape the blobs so that they match the image shape
net.blobs['data'].reshape(2,2,64,64)

images = []
images.append(caffe.io.load_image('./images/dog.jpg'))
images.append(caffe.io.load_image('./images/cat.jpg'))

net.blobs['data'].data[0, ...] = transformer.preprocess('data', images[0])
net.blobs['data'].data[1, ...] = transformer.preprocess('data', images[1])
# net.blobs['data'].data[...] = transformer.preprocess('data', images[1])

# Compute the output
output = net.forward()

print (output)

