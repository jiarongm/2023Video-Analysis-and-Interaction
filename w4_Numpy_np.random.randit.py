# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:26:11 2023

@author: USER
"""
import numpy as np
import cv2
import sys
import random
r=np.random.randint(20,255,size=(300,300),dtype=np.uint8)
g=np.random.randint(20,255,size=(300,300),dtype=np.uint8)
b=np.random.randint(20,255,size=(300,300),dtype=np.uint8)

img2=cv2.merge((b,g,r)) #三種通道都有(彩色)
img1=cv2.add(r,g) #兩矩陣相加(還是一個通道)(灰階)
cv2.imshow('rgb',img2)
cv2.imshow('gray',img1)
cv2.imshow('r',r)
cv2.imshow('g',g)
k=cv2.waitKey(0)
if k==ord("a"):
    cv2.destroyAllWindows( )
