import cv2
import numpy as np

# 開啟攝影機
cap = cv2.VideoCapture(0)

# 設定初始值
lower_skin = np.array([0, 20, 70])
upper_skin = np.array([20, 255, 255])

def update_h(x):
    lower_skin[0] = cv2.getTrackbarPos("Lower H", "image")
    upper_skin[0] = cv2.getTrackbarPos("Upper H", "image")
def update_s(x):
    lower_skin[1] = cv2.getTrackbarPos("Lower S", "image")
    upper_skin[1] = cv2.getTrackbarPos("Upper S", "image")
def update_v(x):
    lower_skin[2] = cv2.getTrackbarPos("Lower V", "image")
    upper_skin[2] = cv2.getTrackbarPos("Upper V", "image")

# 建立視窗
cv2.namedWindow("image")

# 建立Trackbar
cv2.createTrackbar("Lower H","image",0,255,update_h)
cv2.createTrackbar("Upper H","image",20,255,update_h)
cv2.createTrackbar("Lower S","image",20,255,update_s)
cv2.createTrackbar("Upper S","image",255,255,update_s)
cv2.createTrackbar("Lower V","image",70,255,update_v)
cv2.createTrackbar("Upper V","image",255,255,update_v)

while(True):
    # 從攝影機擷取一幀影像
    ret, frame = cap.read()
    
    if ret == True:
        # 將圖片轉換為HSV色域
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 在HSV色域尋找膚色區域
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # 對原圖進行遮罩處理
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # 顯示圖片
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

    # 按下 q 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機資源
cap.release()

# 關閉所有視窗
cv2.destroyAllWindows()