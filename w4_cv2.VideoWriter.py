import cv2
cap=cv2.VideoCapture(0)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
fourcc= cv2.VideoWriter_fourcc(*'mp4v')  # 設定儲存的格式
out=cv2.VideoWriter('01.mp4',fourcc,30.0,(640,480))#產生空的影像檔
capture_time = 20.0 #設定影片時長
start_time = cv2.getTickCount() #開始計時
while True:
    ret,frame=cap.read()
    if not ret:
        print("can't recive frame")
        break
    cv2.imshow('frame',frame)
    out.write(frame) #把frame放入影像檔內
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    #結束時間=(結束-開始)/頻率
    if elapsed_time >= capture_time or cv2.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()