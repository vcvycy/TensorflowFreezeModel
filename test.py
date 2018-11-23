import cv2   
from mtcnn.mtcnn import MTCNN
img = cv2.imread("1.jpg")   
cv2.namedWindow("Image")   
cv2.imshow("Image", img)   
detector = MTCNN()
print(detector.detect_faces(img))
cv2.waitKey (0)  
cv2.destroyAllWindows()  