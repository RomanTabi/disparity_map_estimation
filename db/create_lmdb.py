#!/usr/bin/python
import cv2
import numpy as np

import sys
caffe_root = '../../caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import lmdb

IMAGE_WIDTH = 10
IMAGE_HEIGHT = 10

# Take greyscale image, perform histogram equalization and resize image
# Parameters
#   img - Source image
#   img_width - new image width
#   img_height - new image height
# Returns
#   transformed image
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
  
  img[:, :] = cv2.equalizeHist(img[:, :])

  img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

  return img

# Take images, stack data depth wise and return Datum object containing images data and label
# Parameters
#   img_1 - image 1
#   img_2 - image 2
#   label - {-1, 1} nonmatch (-1) or match (1)
# Returns
#   Datum object containing images data and label
def make_datum(img_1, img_2, label):

  stack_img = np.dstack((img_1, img_2))

  datum = caffe_pb2.Datum()

  np_img_1 = np.array(img_1)
  np_img_2 = np.array(img_2)

  np_img = np.dstack((np_img_1, np_img_2))
  np_img.resize(2, 10, 10)

  datum = caffe.io.array_to_datum(np_img, label)

  return datum
  # return caffe_pb2.Datum(
  #   channels=2, # Two greyscale images, one channel for each
  #   width=IMAGE_WIDTH,
  #   height=IMAGE_HEIGHT,
  #   label=label,
  #   data=img_1.tostring())

##################################################################################################
# Main
# Run histogram equalization. Histogram equalization is a technique for adjusting the contrast of images.
# Resize all training images to a 10x10 format.
# Divide the training data into 2 sets: 
#   One for training (5/6 of images) and the other for validation (1/6 of images). 
# The training set is used to train the model, and the validation set is used to calculate the accuracy of the model.
# Store the training and validation in 2 LMDB databases. train_lmdb for training the model and validation_lmbd for model evaluation.
##################################################################################################

MATCH_FILE = "m50_50000_50000_0.txt"
PATH_TO_MATCH_FILE = "/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/"
PATH_TO_PATCHES = "/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/patches/"

train_lmdb = "/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/train_lmdb_small"
validation_lmdb = "/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/validation_lmdb_small"

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))

correspondences = np.loadtxt(PATH_TO_MATCH_FILE + MATCH_FILE, dtype=int)[:, [0, 1, 3, 4]]

with in_db.begin(write=True) as in_txn:
  for in_idx, correspondence_vector in enumerate(correspondences):

    if in_idx % 6 == 0:
      continue
    
    patch_1_name = correspondence_vector[0]
    patch_2_name = correspondence_vector[2]
    point_1 = correspondence_vector[1]
    point_2 = correspondence_vector[3]

    patch_1 = cv2.imread(PATH_TO_PATCHES + str(patch_1_name) + ".png", cv2.IMREAD_GRAYSCALE)
    patch_2 = cv2.imread(PATH_TO_PATCHES + str(patch_2_name) + ".png", cv2.IMREAD_GRAYSCALE)

    transform_img(patch_1)
    transform_img(patch_2)

    if point_1 == point_2:
      label = 1
    else:
      label = -1

    datum = make_datum(patch_1, patch_2, label)

    in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
    print ('{:0>5d}'.format(in_idx))

    if in_idx >= 1000:
      break;

in_db.close()

print 'Creating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))

correspondences = np.loadtxt(PATH_TO_MATCH_FILE + MATCH_FILE, dtype=int)[:, [0, 1, 3, 4]]

with in_db.begin(write=True) as in_txn:
  for in_idx, correspondence_vector in enumerate(correspondences):

    if in_idx % 6 != 0:
      continue
    
    patch_1_name = correspondence_vector[0]
    patch_2_name = correspondence_vector[2]
    point_1 = correspondence_vector[1]
    point_2 = correspondence_vector[3]

    patch_1 = cv2.imread(PATH_TO_PATCHES + str(patch_1_name) + ".png", cv2.IMREAD_GRAYSCALE)
    patch_2 = cv2.imread(PATH_TO_PATCHES + str(patch_2_name) + ".png", cv2.IMREAD_GRAYSCALE)

    transform_img(patch_1)
    transform_img(patch_2)

    if point_1 == point_2:
      label = 1
    else:
      label = 0

    datum = make_datum(patch_1, patch_2, label)

    in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
    print ('{:0>5d}'.format(in_idx))

    if in_idx >= 1000:
      break;

in_db.close()
