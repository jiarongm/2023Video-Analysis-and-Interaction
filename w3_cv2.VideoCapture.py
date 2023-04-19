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
    if cv2.waitKey(1)==ord('s'):
        print("----Start to save video.---\n")
        fourcc= cv2.VideoWriter_fourcc('DIVX')  # 設定儲存的格式
        cv2.VideoWriter('save_video',fourcc,20.0,(680,480),1)
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
