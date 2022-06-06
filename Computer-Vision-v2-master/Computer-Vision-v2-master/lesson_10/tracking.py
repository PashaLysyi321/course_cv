#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:48:46 2021

@author: janko

This script demonstrate the block matching tracking approach.

You can find the dataset here:
    https://www.kaggle.com/soumikrakshit/udacity-car-dataset-crowdai

"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the dataset
folder = '/home/janko/Projects/robot_dreams/cv/data/datasets/object-detection-crowdai'
frames = os.listdir(folder)

# Sort (alphabetically) to ensure temporal consecutiveness
frames.sort()
idx = frames.index('1479498704988166448.jpg')

# Let's assume the detector has detected two vehicles with the following bounding boxes
obj_id = 0
if obj_id == 0:
    x1, y1 = 910, 545
    x2, y2 = 1025, 655
elif obj_id == 1:
    x1, y1 = 680, 540
    x2, y2 = 815, 660
    
width = x2 - x1
height = y2 - y1

# Limit the search to a certain vicinity (since the cars can only move that fast)
search = 50

# Genrate tracking template
img = cv2.imread(os.path.join(folder, frames[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
template = img[y1:y2, x1:x2]/255

# Tracking loop
for ii in range(idx, idx + 10):
    img = cv2.imread(os.path.join(folder, frames[ii]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Build local search window
    search_window = img[y1-search:y2+search, x1-search:x2+search]/255    
    
    # Tracking by minimising simple SAD (sum of absolute differences) loss
    # Equivalent to MSE loss (in **this** case) but faster
    track_x1 = None
    track_y1 = None
    loss = 1e6
    for r in range(0, search_window.shape[0] - height):
        for c in range(0, search_window.shape[1] - width):
            candidate = search_window[r:r+height, c:c+width]
            score = np.sum(np.abs(template - candidate))
            if score < loss:
                loss = score
                track_x1 = c
                track_y1 = r                
                    
    # Update the bounding box of the tracked object
    x1 = x1 - search + track_x1
    y1 = y1 - search + track_y1
    print(x1, y1, width, height)
            
    # Show the tracker working
    cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 5)
    plt.imshow(img)
    plt.show(), plt.draw()
    plt.waitforbuttonpress()
    plt.clf()
    
    # fname = 'frame_' + str(ii).zfill(3) + '.jpg'
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join('/home/janko/Projects/robot_dreams/cv/data/tracking/sad', fname), img)