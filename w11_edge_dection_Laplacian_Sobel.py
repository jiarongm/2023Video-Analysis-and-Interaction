# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:25:40 2023

@author: USER
"""

import numpy as np
import cv2 

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    lo=cv2.Laplacian(frame,cv2.CV_64F)
    sobelx=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    cv2.imshow('frame',frame)
    cv2.imshow("lo",lo)
    cv2.imshow("sx",sobelx)
    cv2.imshow("sy",sobely)
    
    if cv2.waitKey(1)==ord('s'):
        break
cv2.destroyAllWindows
cap.release()