# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:05:57 2023

@author: USER
"""

import numpy as np
import cv2
img1=cv2.imread("test1.png")#圖片大小要一樣
img2=cv2.imread("test2.png")
i=0.00
while True:
    i=i+0.01
    dst=cv2.addWeighted(img1,i,img2,1-i,50)
    cv2.imshow("dst",dst)
    print("a=",i)
    k=cv2.waitKey(10)#等待時間1000:1秒
    if k==ord('s') or i>=1:
        cv2.destroyAllWindows()#最後關掉所有視窗
        break