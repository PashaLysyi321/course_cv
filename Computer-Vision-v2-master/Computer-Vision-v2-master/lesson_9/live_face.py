#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script runs the dlib face detector on live video capture from computer camera.

Created on Sat Feb 12 17:04:41 2022

@author: janko
"""

# import the opencv library
import cv2
import dlib

# Let's load the detector
detector = dlib.get_frontal_face_detector()

def rect_to_bb(rect):
    # Dlib rect --> OpenCV rect
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
    
# Define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    rects = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)
    
    # Draw rectangle around each face        
    for rect in rects:            
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)        
      
    cv2.imshow('frame', frame)
      
    # The 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object and destroy all the windows
vid.release()
cv2.destroyAllWindows()