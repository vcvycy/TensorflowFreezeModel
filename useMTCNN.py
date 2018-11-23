from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import detect_face
import cv2
import random
def drasPos(img,pos):
  for i in range(5):
    x=pos[i]
    y=pos[i+5]
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(img,(x,y),(x,y),color,1)
    
def draw(img,bounding_boxes,pos):
  #box
  margin=0
  for b in bounding_boxes:
    x1=int(b[0])-margin
    y1=int(b[1])-margin
    x2=int(b[2])+margin
    y2=int(b[3])+margin
    #print("(%s,%s)" %((x1,y1),(x2,y2)))
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
  #face pos
  #drasPos(img,pos) 
def loadFromNPY():
    print('Creating networks and loading parameters') 
    sess = tf.Session() 
    return detect_face.create_mtcnn(sess, None)
    
def show(img):
    b,g,r=cv2.split(img)
    img=cv2.merge([r,g,b])
    cv2.namedWindow("cjf", cv2.WINDOW_FREERATIO)
    cv2.imshow("cjf",img)
    cv2.waitKey(0)

if __name__=="__main__":
    pnet, rnet, onet = loadFromNPY()
    print("[*]MTCNN load success")
    #parameters
    minsize = 35 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    #img_list=["ttt.jpg"]
    #img_list=["gyy.jpeg"]
    img_list=["2.jpg"]#,"ttt.jpg"] 
    for image in img_list: 
        img=misc.imread(image,mode="RGB") 
        bounding_boxes, pos = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        draw(img,bounding_boxes,pos)  
        show(img)
        #print("\n[*] bounding boxes:%s" %(bounding_boxes)) 
        
        
        