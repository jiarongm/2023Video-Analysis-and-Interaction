# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:36:08 2023

@author: USER
"""

import numpy  as np
import cv2
from matplotlib import pyplot as plt

# 讀取256灰階影像
img = cv2.imread('256gray.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('256gray.jpg')
if img is None:
    print('Failed to load image!')
else:
    print('Image loaded!')

# 將256灰階轉換為16 bin的灰階影像
bin_num = 16
bins = np.linspace(0, 255, bin_num+1)
bin_idx = np.digitize(img.ravel(), bins)
img_16bin = np.zeros_like(img)
for i in range(bin_num):
    img_16bin[bin_idx == i] = i * (255//bin_num)

# 顯示轉換後的影像
cv2.imshow('16-bin Gray Scale Image', img_16bin)

# 顯示影像直方圖
plt.hist(img_16bin.ravel(), bins=bin_num)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()