#!/usr/bin/python
print 'Disparity Map Estimation';

import cv2
from PIL import Image
import matplotlib.pyplot as plt
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
if os.path.isfile('model/yosemite.caffemodel'):
    print 'CaffeNet found.'

caffe.set_mode_cpu()

model_def = './estimation_model_deploy.prototxt'
model_weights = './snapshots/_iter_40000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
# transformer.set_transpose('data', (1))  # move image channels to outermost dimension
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_channel_swap('data', (0, 1))  # swap channels from RGB to BGR

# We need to reshape the blobs so that they match the image shape
net.blobs['data'].reshape(1,2,10,10)

image_path_1 = sys.argv[1]
image_path_2 = sys.argv[2]

images = []
images.append(caffe.io.load_image(image_path_1))
images.append(caffe.io.load_image(image_path_2))


print net.blobs['data'].data[0, 0, ...].size

# .transpose(2, 0, 1).reshape(3, 64, 64)
# Shape of X
# print (x.shape)
# with open('test_origin.txt','wb') as f:
  # np.savetxt(f, x,fmt='%s')

  
# Image je 64x64x3 -> potrebujeme 64x64x1 (64x64) -> tie 3 kanaly obsahuju tie iste cisla -> staci jeden
# Resize image 1 -> reduce channels
img_1 = np.array(images[0], order='F')
img_1.resize((10, 10))
print (img_1.shape)
with open('test_resize.txt','wb') as f:
  np.savetxt(f, img_1,fmt='%s')

# Resize image 2 -> reduce channels
img_2 = np.array(images[1], order='F')
img_2.resize((10, 10))
print (img_2.shape)


print images[0][0].size
print images[0][0][0].size

net.blobs['data'].data[0, 0, ...] = transformer.preprocess('data', img_1)
net.blobs['data'].data[0, 1, ...] = transformer.preprocess('data', img_2)
# net.blobs['data'].data[...] = transformer.preprocess('data', images[1])

print 'arg 1: ', sys.argv[1]
print 'arg 2: ', sys.argv[2]



# Compute the output
output_10 = net.forward(end='caffe.SpatialConvolution_10')
output_13 = net.forward(end='caffe.SpatialConvolution_13')
# output_16 = net.forward(end='caffe.SpatialConvolution_16')
output_18 = net.forward(end='caffe.Flatten_18')


print ('SpatialConvolution_10 = ', output_10['caffe.SpatialConvolution_10'].size)
print ('SpatialConvolution_13 = ', output_13['caffe.SpatialConvolution_13'].size)
print ('Flatten_18 = ', output_18['caffe.Flatten_18'].size)

output = net.forward()
print (output)

# print 'Difference = ', abs(output['caffe.InnerProduct_22'][0] - output['caffe.InnerProduct_22'][1])

