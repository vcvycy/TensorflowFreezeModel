import cv2
import numpy as np
img=cv2.imread("ttt.bmp")
img2=np.flipud(img)
print("type:%s" %([img2.shape]))
cv2.namedWindow("cjf")
cv2.imshow("cjf",img2)
cv2.waitKey(0)