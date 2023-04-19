import numpy as np
import cv2

global img

		

filename = input( "Please enter filename: " )
img = cv2.imread( filename, -1 )
cv2.namedWindow( filename )
#cv2.setMouseCallback( filename, onMouse )
#cv2.imshow( filename, img )
x=input( "Please enter X: " )
y=input( "Please enter Y: " )
for i in range(20):
    for j in range(40):
        img[int(x)+i,int(y)+j]=[0,0,0]
cv2.imshow( filename, img )
k=cv2.waitKey(0)
if k==ord("a"):
    cv2.destroyAllWindows( )
