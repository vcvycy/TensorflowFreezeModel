"""
  测试mtcnn_freezed_model.pb是否可用
"""
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
from tensorflow.python.framework.graph_util import convert_variables_to_constants
img_size=40
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  
def loadImage(path):
  img=misc.imread(path)
  img=misc.imresize(img,(img_size,img_size),interp='bilinear')
  return img
  #return prewhiten(img)
if __name__=="__main__":
   #load pb
   with open("mtcnn_freezed_model.pb","rb") as f:
    #(1)load model from protocl buffer
    sess=tf.Session()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read()) 
    tf.import_graph_def(graph_def) 
    #(2)save tensorboard
    tf.summary.FileWriter("./tboard",sess.graph)
    #(3) PNET
    minsize = 38 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0 # scale factor
    
    p_t_input =tf.get_default_graph().get_tensor_by_name("import/pnet/input:0")
    p_t_prob  =tf.get_default_graph().get_tensor_by_name("import/pnet/prob1:0")
    p_t_bbr   =tf.get_default_graph().get_tensor_by_name("import/pnet/conv4-2/BiasAdd:0")
    img=[misc.imread("timg.jpg",mode="RGB")]
    p_prob,p_bbr=sess.run((p_t_prob,p_t_bbr),feed_dict={p_t_input:img}) 