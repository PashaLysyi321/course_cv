#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:23:59 2021

This script demonstrate the use of OpenCV trackers.

You can find the dataset here:
    https://www.kaggle.com/soumikrakshit/udacity-car-dataset-crowdai

@author: janko
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
idx = frames.index('1479498995506866553.jpg')

# Let's assume the detector has detected a vehicle
x1, y1 = 965, 555
x2, y2 = 1015, 595


width = x2 - x1
height = y2 - y1

# Limit the search to a certain vicinity (since the cars can only move that fast)
search = 50

# Set up tracker
tracker_types = ['MIL','KCF', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()

if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()

if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# Genrate tracking template
img = cv2.imread(os.path.join(folder, frames[idx]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize tracker
bbox = (x1, y1, width, height)
ok = tracker.init(img, bbox)


# Tracking loop
for ii in range(idx, idx + 50):
    img = cv2.imread(os.path.join(folder, frames[ii]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        
    ok, bbox = tracker.update(img)
    print(ok, bbox)

            
    # Show the tracker working
    x1, y1 = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    cv2.rectangle(img, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show(), plt.draw()    
    plt.waitforbuttonpress(0.1)
    plt.clf()
    
    # boxes.append([x1, y1, width, height])
    # with open('/home/janko/Projects/robot_dreams/cv/data/tracking/csrt/boxes.pckl', 'wb') as fid:
    #     pickle.dump(boxes, fid)
        
    