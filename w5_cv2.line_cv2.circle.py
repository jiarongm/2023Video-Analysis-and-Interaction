# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:57:18 2023

@author: USER
"""

import cv2
import numpy as np
img1=cv2.imread("test1.png")
x,y,z=img1.shape
print(x,y)
cv2.line(img1 ,(0,0),(y-1,x-1),(0,0,255),5)
cv2.line(img1 ,(0,x),(y,0),(0,0,255),5)
cv2.circle(img1,(int(y/2),int(x/2)),100,(0,255,0),-1)
cv2.imshow("draw",img1)
k=cv2.waitKey(0)
cv2.destroyAllWindows()