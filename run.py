#!/usr/bin/python
import os
import sys
import cv2
from PIL import Image
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

HEIGHT = 0
WIDTH = 1

if len(sys.argv) != 5 :
  print "run img_1 img_2 patch_width patch_height"
  sys.exit()

img_1_path = sys.argv[1]
img_2_path = sys.argv[2]
patch_width = int(sys.argv[3])
patch_height = int(sys.argv[4])

offset_x = 0
offset_y = 660
# Regions of interest
img_1 = cv2.imread(img_1_path)
img_2 = cv2.imread(img_2_path)

while offset_x + patch_width < img_1.shape[WIDTH] :
  # Crop images ( img[y: y + h, x: x + w] )
  crop_img_1 = img_1[offset_y : offset_y + patch_height, offset_x : offset_x + patch_width]
  crop_img_2 = img_2[offset_y : offset_y + patch_height, offset_x : offset_x + patch_width]
  # Show images
  plot_img = np.concatenate((crop_img_1, crop_img_2), axis=1)
  cv2.imshow("cropped_1", plot_img)
  # cv2.imshow("cropped_2", crop_img_2)

  g1 = cv2.cvtColor(crop_img_1, cv2.COLOR_BGR2GRAY );
  g2 = cv2.cvtColor(crop_img_2, cv2.COLOR_BGR2GRAY );

  cv2.imwrite("./tmp/1.png", g1);
  cv2.imwrite("./tmp/2.png", g2);

  cv2.waitKey(0)

  offset_x += patch_width
