#!/usr/bin/python
import cv2

DATA_PATH = '/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/'
OUTPUT_DATA_PATH = '/media/roman/887CE0237CE00DAE/ubuntu_downloads/liberty/patches/'
IMAGE_NAME_PREFIX = "patches"
IMAGE_NAME_SUFFIX = ".bmp"
PATCH_NAME_SUFFIX = ".png"

TOTAL_PATCHES = 450092
NUMBER_OF_IMAGES = 1759
PATCH_SIZE = 64
IMAGE_SIZE = 16 # Number of patches in single dimension

patch_number = 0

for i in range(0, NUMBER_OF_IMAGES):
  image_name = DATA_PATH + IMAGE_NAME_PREFIX + "%04d" % i + IMAGE_NAME_SUFFIX
  
  print ("Image: " + image_name)

  img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

  for row in range(0, IMAGE_SIZE):
    for col in range(0, IMAGE_SIZE):
      offset_y = row * PATCH_SIZE
      offset_x = col * PATCH_SIZE
      # Crop images ( img[y: y + h, x: x + w] )
      patch = img[offset_y : offset_y + PATCH_SIZE, offset_x : offset_x + PATCH_SIZE]

      patch_name = str(patch_number) + PATCH_NAME_SUFFIX
      cv2.imwrite(OUTPUT_DATA_PATH + patch_name, patch);

      patch_number += 1

      if patch_number == TOTAL_PATCHES:
        break