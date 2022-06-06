#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:19:35 2021

@author: janko

This script demonstrate the problems of block matching tracking approach.
For object with changing apperance (e.g. the pixel size that depends on distance to camera),
the BM approach tend to lose the track.

You can find the dataset here:
    https://www.kaggle.com/soumikrakshit/udacity-car-dataset-crowdai
    
"""

import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Load the dataset
folder = '/home/janko/Projects/robot_dreams/cv/data/datasets/object-detection-crowdai'
frames = os.listdir(folder)

boxes = []

# Sort (alphabetically) to ensure temporal consecutiveness
frames.sort()
# idx = frames.index('1479498379965419997.jpg')
idx = frames.index('1479498388466168072.jpg')

idx = frames.index('1479498406467355722.jpg')

idx = frames.index('1479498995506866553.jpg')

# Let's assume the detector has detected a vehicle
# x1, y1 = 1060, 580
# x2, y2 = 1120, 630

x1, y1 = 935, 555
x2, y2 = 1080, 675

x1, y1 = 970, 570
x2, y2 = 1100, 670

x1, y1 = 965, 555
x2, y2 = 1015, 595

width = x2 - x1
height = y2 - y1

# Limit the search to a certain vicinity (since the cars can only move that fast)
search = 50

# Genrate tracking template
img = cv2.imread(os.path.join(folder, frames[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
template = img[y1:y2, x1:x2]/255

cnt = 0
# Tracking loop
for ii in range(idx, idx + 50):
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
    
    # fname = 'frame_' + str(cnt).zfill(3) + '.jpg'
    # cv2.imwrite(os.path.join('/home/janko/Projects/robot_dreams/cv/data/tracking/frames', fname),
    #             cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cnt = cnt + 1
            
    # Show the tracker working
    cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
    plt.imshow(img)    
    
    plt.show(), plt.draw()    
    plt.waitforbuttonpress(0.1)
    plt.clf()
    
    # boxes.append([x1, y1, width, height])
    # with open('/home/janko/Projects/robot_dreams/cv/data/tracking/sad/boxes.pckl', 'wb') as fid:
    #     pickle.dump(boxes, fid)