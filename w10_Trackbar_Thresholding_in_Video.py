import numpy as np
import cv2
import sys
cap=cv2.VideoCapture(0)

def nothing(x):
    pass
cv2.namedWindow('BW')
cv2.createTrackbar('Trackbar1', 'BW', 100, 255, nothing)
while True:
    ret,frame=cap.read()
    if not ret:
        print("can't recive frame")
        break
    cv2.imshow('Original',frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)

    thrs1 = cv2.getTrackbarPos('Trackbar1', 'BW')
    ret, balck_and_white= cv2.threshold(gray,thrs1,255,cv2.THRESH_BINARY)
    cv2.imshow('BW', balck_and_white)
    ch = cv2.waitKey(1)
    if ch == 27:   #ESC keystroke
    	break
cap.release()
cv2.destroyAllWindows()
