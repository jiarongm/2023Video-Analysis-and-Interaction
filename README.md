# 視訊分析
[TOC]
## week2_introduction
Digitization數位化:用3步驟，將float像素值數位化為數位影像
1. Sampling : 將時間軸離散化
2. **==Quantization 量化==** : 將振幅軸切割成有限的位階(level)，將 **振幅軸離散化，浮點數變成整數**
3. Coding : 整數值編成二元碼

物體經由鏡片投影產生影像(連續的)，
![](https://i.imgur.com/WbICvcD.png)
生成取樣影像(像素位置是離散的、色彩值是連續的)，量化將色彩值轉為離散數值
![](https://i.imgur.com/M1Zw3Re.png)

取樣(產生離散的位置x,y)+量化(每個位置得到離散的色彩值0~255)=數位影像可表示為二維矩陣，矩陣中的每一個元素稱為 **像素 I(x,y)**
MxN矩陣>> x=0~M-1;y=0~N-1 有M個row + N個colum

**數位影像的品質:**
* 取樣決定空間解析度
* 量化決定色彩解析度
    R,G,B = 分別為256 level(8 bit)

**數位影像的類別:由量化(bit)數決定**
* Binary 二值影像(1 bit):0 or 1
* Gray-level(1 byte = 8bit):0~255
* True-Color(3 byte):255^3=16.7M colors
* Indexed-color(log2n bit)索引全彩影像，用較少的bit表示彩色影像
    ![](https://i.imgur.com/mlw94cE.png)

---
## week3_Image_Read、Show、Write
* **```cv2.imread()```** 讀
    同個目錄下'name.jpg';其他目錄:'檔案路徑';0:gray imge/1:color/-1:alpha channel
* **```cv2.imshow()```** 秀
* **```cv2.imwrite()```** 存
```p=
import cv2    //使用openCV
img=cv2.("pohot_name",-1)    // -1:用灰階 讀入圖片(放在同目錄下)放到陣列img裡
cv2.imshow("wondows_name",img)    //開視窗 秀圖片img
cv2.waitkey(0)    //等待輸入的指令(字母)/延遲X秒後結束 0:無限秒
cv2.destroyAllWindows()     //關閉所有視窗
```
**```cv2.waitKey(delay)```** 接受一個按鈕事件並返回按鈕的ASCII碼; ==delay=1000:延遲1秒== 0:延遲時間無窮大
**```cv2.destroyWindow(winname)```** 關閉某視窗
e.g.
![](https://i.imgur.com/rz9dzKf.png)
>超過0~255範圍 會不自然

圖片的型態```img.shape``` :秀出圖像的長垂直、寬水平、色彩(灰階1 黑白2 彩色3)
```python!
print("color_img_shap",img.shap)
```
![](https://i.imgur.com/SWS9ifa.png)


 **```cv2.imwrite("outname.jpg",img)```** 存檔，搭配 ```cv2.waitkey(0)```使用-按某鍵存檔:
 ```p=
 k=cv2.waitkey(0)
 ...
 if k==ord('s')    //ord('c')==char()一個字母 回傳ASCII碼
     cv2.imwrite("outname.jpg",img)
 ```

### 單獨秀出某通道的圖像

 Python openCV uses **BGR** mode 顯示各個通道的圖像
* **```b,g,r=cv2.split(img)```** 把3種通道分割開來
* **```img2=cv2.merge(b,g,r)```** 合併成3種通道(彩色影像)
* ```cv2.imgshow()```

先切割BGR的通道>>有三個 長*寬 的矩陣，**裡面分別放0~255的數字**
**單獨抓某個通道會是灰階圖片**
```python!
cv2.imshow('single_green_channel', g)
#display single channel (regard as gray-level image)
```
所以要把其他兩個通道清空(放0)，再與某通道合併，才能呈現單獨某色的圖像
```python=
img = cv2.imread( "WIN_20230302_13_18_26_Pro.jpg")
b,g,r = cv2.split(img)    #分割成3種通道b,g,r
if img is None:
    sys.exit("could not read photo.")
cv2.imshow( "Example", img )

b[:]=0    #b矩陣清空放0
g[:]=0
r_color=cv2.merge((b,g,r))

k=cv2.waitKey( 0 )
if k==ord("s"):
    cv2.imshow( "R_color",r_color )
    k=cv2.waitKey( 0 )
if k==ord("a"):
    cv2.destroyAllWindows( )
print("color_img_shap",img.shape)
cv2.destroyAllWindows( )
print("done.")
```
---
## week4_video
**```cap=cv2.VideoCapture(0)```** create攝影機物件object
* 0:預設第一隻 1:第二隻 2:第三隻
* 一幀一幀(frame by frame)的捕捉攝像頭

**```ret,frame=cap.read()```** 回傳兩個值(bool/numpy.array)
* ret:True正確取幀(沒有漏掉)或 False 
* frame:下一張影像的矩陣

**```cap.release() ```** 釋放幀;關閉攝影機


```cap.isOpened()```檢查攝影機是否啟動
```cap.open()```開啟攝影機

```cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)```將影像轉成灰階
```python=
#HW22目的:用攝影機捕捉1.全彩視訊、2.紅色、3.藍綠交換
cap=cv2.VideoCapture(0)
while True:    #在無窮迴圈裡一直秀圖片>>影片串流
    ret,frame=cap.read()
    if not ret:    #如果是flase(不正常取幀)
        print("can't recive frame")
        break
    cv2.imshow('frame',frame)    #秀出全彩影片串流
    
    b,g,r=cv2.split(frame)    #分割三種通道
    temp=b    #把blue green通道交換
    b=g
    g=temp
    exchange=cv2.merge((b,g,r))    #合併
    cv2.imshow('exchange',exchange)

    b[:]=0    #清空(秀紅色通道)
    g[:]=0
    r_color=cv2.merge((b,g,r))
    cv2.imshow('R_color',r_color)
    
    
    if cv2.waitKey(1)==ord('q'):    #按Q結束
        cap.release()    #釋放幀
        cv2.destroyAllWindows()
        break

```
### 儲存影片 
**```cv2.VideoWriter(output_name,int fourcc,fps,size,全彩:true or flase)```**
* 檔名
* 壓縮的方式compress the frames: **DIVX**(Windows建議使用) 、XVID..
* fps每秒幀數
* 寬width、高heigt ex.640*480
* true:color false:gray

```p=
fourcc= cv2.VideoWriter_fourcc('DIVX')// 設定儲存的格式
out=cv2.VideoWriter(output_name,fourcc,20.0,(680,480))
```
### Numpy
![](https://i.imgur.com/rlOzXKZ.png)
```
x=np.array([ [1,2,3],[4,5,6] ])
```
#### 陣列屬性:
* np.ndim維度
* np.shap外型 #(row,column)
* np.size大小
* np.dtype資料內容:(unit8)

#### 陣列+-* /運算:  **element-wise:*/是一個元素對一個元素運算**
* **```np.multiply()```** 一個一個相乘
    ![](https://i.imgur.com/OyOscZz.png)
    ![](https://i.imgur.com/BdbGvuu.png)

* **```C=np.transpose (B)```** 一般矩陣乘法:要先轉置 讓基底相同
    **```result_1= np.matmul (A , C)```** 矩陣相乘

元素都是0的矩陣 ```np.zeros(shape,dtype =float)```
元素都是1的矩陣 ```np.ones(shape,dtype =float)```

隨機陣列 ==**```np.random.randint(low最小值(含),high(不含),size維度,dtype=int)```**==
```python=
#創造隨機矩陣，隨機給r、g、b的值，單獨秀出r、g灰階影像與三種通道的影像
import numpy as np
import cv2
import sys
import random
r=np.random.randint(20,255,size=(300,300),dtype=np.uint8)
g=np.random.randint(20,255,size=(300,300),dtype=np.uint8)
b=np.random.randint(20,255,size=(300,300),dtype=np.uint8) //uint8 沒有符號int 8bit

img2=cv2.merge((b,g,r)) #三種通道都有(彩色)
img1=cv2.add(r,g) #兩矩陣相加(還是一個通道)(灰階)
cv2.imshow('rgb',img2)
cv2.imshow('gray',img1)
cv2.imshow('r',r)
cv2.imshow('g',g)
k=cv2.waitKey(0)
if k==ord("a"):        #按A結束
    cv2.destroyAllWindows( )
```

e.g. 20x40的黑色區域
![](https://i.imgur.com/qMCJtUj.jpg)

```python=
import numpy as np
import cv2
global img

filename = input( "Please enter filename: " ) #輸入圖片檔名
img = cv2.imread( filename, -1 )
cv2.namedWindow( filename )
#cv2.setMouseCallback( filename, onMouse ) 用滑鼠
cv2.imshow( filename, img )
x=input( "Please enter X: " )
y=input( "Please enter Y: " )
for i in range(20):            #用雙層迴圈 i:row的數量 南北向
    for j in range(40):        #j:colum數 東西向
        img[int(x)+i,int(y)+j]=[0,0,0]
cv2.imshow( filename, img )
k=cv2.waitKey(0)
if k==ord("a"):
    cv2.destroyAllWindows( )
```
**HW23.用內建的相機捕捉20秒的影片並儲存**.
> cv2.getTickCount ()函數向我們返回從參考事件發送到調用cv2.getTickCount()函數的時間的時鐘信號計數。
> 
> 要計算代碼塊執行的時間，我們可以使用cv2.getTickCount()在代碼塊的開頭和結尾計算時鐘信號的數量，並將其差值除以頻率，可以使用cv2.getTickFrequency ()函數。


==**```cv2.getTickCount()```**== 計算兩次getTickCount()之間的時鐘數量/頻率 **```cv2.getTickFrequency ()```** =時間
```python=
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
#還是無法固定影片儲存的時長，可能被執行時間/電腦效能影響(會越跑越短)
```
e.g.呈上，再秀出b、g、r三種通道的影像
==**```np.zeros_like(matrix)```**== 創造與matrix相同形狀的矩陣，但元素皆是0
**```cap.get(cv2.CAP_PROP_FRAME_WIDTH)```** 捕捉相機長寬
```python=
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
    cv2.imshow('frame',frame)#秀出來
    cv2.imshow('b_color',b_color)
    cv2.imshow('g_color',g_color)
    cv2.imshow('r_color',r_color)
    key=cv2.waitKey(1)
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    #結束時間=(結束-開始)/頻率
    if elapsed_time >= capture_time or  key== ord('q'):
        break
cap.release()#釋放資源
out_color.release()
out_g.release()
out_b.release()
out_r.release()
cv2.destroyAllWindows()
```

---
## week5_OpenCV基礎影像操作
**ROI:region-of-interest**
```img[200:300,400:500]``` >>> X:200~300 Y:400~500 之間的範圍
### image blending影像融合
**```cv2.addWeighted(img1,a,img2,i-a,gamma)```** 影像融合(兩張圖片大小要相同)

``` python=
#影像融合 用a透明度讓影像漸變到另一張
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
```
### 在影像上畫圖
* **```cv2.line()```**
    - ( img,(x1,y1),(x2.y2),(B,G,R),Thickness粗細 )
* **```cv2.circle()```**,y
    - -1:填滿
* **```cv2.rectangle()```** 矩形
* **```cv2.ellipse()```**  橢圓
* **```cv2.putText()```**

```python=
#HW33圖中心畫綠色圓+對角紅色線
import cv2
import numpy as np
img1=cv2.imread("test1.png")
x,y,z=img1.shape#影像大小 垂直高度/水平寬度/通道數量
print(x,y)
cv2.line(img1 ,(0,0),(y-1,x-1),(0,0,255),5)
cv2.line(img1 ,(0,x),(y,0),(0,0,255),5)
cv2.circle(img1,(int(y/2),int(x/2)),100,(0,255,0),-1) #(int(y/2):避免不能整除產生小數點
cv2.imshow("draw",img1)
k=cv2.waitKey(0)
cv2.destroyAllWindows()
```
### Color Space色彩空間
* RGB:亮度彩度混在一起
* **HSV**:
    - 彩度Hue
    - 飽和度Saturation:how far is the color from gray
    - 亮度Value
    - ![](https://i.imgur.com/D631FFq.png)

* YCrCb

**```cv2.createTrackbar('textname','windows_name',min,max,Fn)```** 滑桿 滑動時執行Fn程式
==**```cv2.bitwise_and(frame,frame,msak=mask)```**== 
* 做bit_and運算
* mask=mask:要提取的範圍
* frame :執行運算的原始影像
```python=
import cv2 as cv2
def image_and(image,mask):    #輸入影像和遮罩
    area = cv2.bitwise_and(image,image,mask=mask) #不打mask的話就是算兩張圖片的交集
    cv2.imshow("area",area)
    return area
```
==**```cv2.inRange(img,lowerb,upperb)```**== **抓取特定範圍的顏色**
* img輸入的影像
* 色彩範圍最低數值
* 色彩範圍最高數值
```python=
#捕捉影片中的藍色部分
import cv2
import numpy as np
cap = cv2.VideoCapture('blue_object_video.mp4')
while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#轉成hsv
    
    lower_blue = np.array([110, 50, 50])#設定遮罩範圍(已知HSV中藍色的範圍
    high_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, high_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

[TOC]
## week6_Scaling、
### 縮放


## week7
### 平移Translation
沿(x,y)方向移動:移動量(tx,ty)
1. 用numpy建構M矩陣(變換矩陣translate matrix)
    * **```np.float32([ [1,0,100],[0,1,50] ])```**
    * ![](https://i.imgur.com/ukdWZHY.png)
2. 傳給 ==**```cv2.warpAffine(img,M,(clos,rows) )```**== 執行變換
    * (cols,rows)輸出影像的大小(width水平寬,height垂直高)
```python=
import cv2
import numpy as np
img=cv2.imread("img_name.jpg",0)
rows,cols=img.shape #用img.shape取得影像大小(垂直高/水平寬/通道)
M=np.float32([ [1,0,100],[0,1,50] ])    #創造變換矩陣M，移動量:(100,50)
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 旋轉Rotation
1. 旋轉角度Θ，使用旋轉矩陣
    * ![](https://i.imgur.com/47HAdlh.png)
2. **```cv2.getRotationMatrix2D( (x,y),Θ,比例)```** 
    * 旋轉中心(x,y)
    * 旋轉角度:逆時針為正、順時針為負
    * 比例
```python=
import cv2
import numpy as np
img=cv2.imread("img_neme.jpg",0)
rows,cols=img.shape[0:2] #用img.shape取得影像大小(垂直高/水平寬/通道)
M=cv2.getRotationMatrix2D( (cols/2,rows/2),90,1)    #創造旋轉矩陣，
dst=cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 剪切Shear
將圖像在在水平垂直方向剪切，實現扭曲和變形

### 直方圖

np.reval()降維:多維降成1維
