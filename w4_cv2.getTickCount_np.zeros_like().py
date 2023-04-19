#加分題09161141 第四組 江佳蓉
#用內建相機分別捕捉(全彩/R/G/B)四種通道並儲存20秒的影片
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
fourcc= cv2.VideoWriter_fourcc(*'mp4v')  # 設定儲存的格式
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))#用cap.get()捕捉攝影機的規格
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_color=cv2.VideoWriter('01_merge.mp4',fourcc,30.0,(width,height))#產生空的影像檔
out_g=cv2.VideoWriter('01_g.mp4',fourcc,30.0,(width,height))
out_b=cv2.VideoWriter('01_b.mp4',fourcc,30.0,(width,height))
out_r=cv2.VideoWriter('01_r.mp4',fourcc,30.0,(width,height))
capture_time = 20.0 #設定影片時長
start_time = cv2.getTickCount() #開始計時

while True:
    ret,frame=cap.read()
    if not ret:
        print("can't recive frame")
        break

    b,g,r=cv2.split(frame)#切割成三種通道 
    matrix=np.zeros_like(b)#創造一個跟b大小相同、元素都是0的矩陣
    r_color=cv2.merge((matrix,matrix,r))#合併
    g_color=cv2.merge((matrix,g,matrix))
    b_color=cv2.merge((b,matrix,matrix))

    out_color.write(frame) #把frame放入影像檔內
    out_g.write(g_color)
    out_r.write(r_color)
    out_b.write(b_color)
    
    cv2.imshow('frame',frame)
    cv2.imshow('b_color',b_color)
    cv2.imshow('g_color',g_color)
    cv2.imshow('r_color',r_color)
    key=cv2.waitKey(1)
    
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    #結束時間=(結束-開始)/頻率
    if elapsed_time >= capture_time or  key== ord('q'):
        break
cap.release()
out_color.release()
out_g.release()
out_b.release()
out_r.release()
cv2.destroyAllWindows()