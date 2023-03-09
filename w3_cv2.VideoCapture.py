import numpy as np
import cv2
import sys

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        print("can't recive frame")
        break
    cv2.imshow('frame',frame)
    
    b,g,r=cv2.split(frame)
    temp=b
    b=g
    g=temp
    exchange=cv2.merge((b,g,r))
    cv2.imshow('exchange',exchange)

    b[:]=0
    g[:]=0
    r_color=cv2.merge((b,g,r))
    cv2.imshow('R_color',r_color)
    
    
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destoryAllWindows()