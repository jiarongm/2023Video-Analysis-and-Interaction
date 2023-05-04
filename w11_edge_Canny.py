# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:15:12 2023

@author: USER
"""

import cv2
cap = cv2.VideoCapture(0)

# 設定攝影機為640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def Fn():
    print("good")
x=50
y=200

cv2.namedWindow('frame')
cv2.createTrackbar("x","frame",40,80,Fn)
cv2.createTrackbar("y","frame",100,200,Fn)
while True:
    ret, frame = cap.read()

    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊降噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 邊緣檢測
    edges = cv2.Canny(blur, x, y)
    x=cv2.getTrackbarPos('x',"frame")
    y=cv2.getTrackbarPos('y',"frame")

    cv2.imshow('frame', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()