# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:52:01 2023

@author: USER
"""
import cv2
def Fuc():
    print("helloword")
cv2.namedWindow("image", 0)
cv2.createTrackbar('s','image',1,10,Fuc)
cv2.setTrackbarPos('s','image',2)
cv2.waitKey(0)
cv2.destroyAllWindows()