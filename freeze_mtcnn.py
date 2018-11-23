"""
  使用convert_variables_to_constants:
  npy模型转换为pb模型
  保存为./mtcnn_freezed_model.pb
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
def freeze(sess,):
    output_name=['pnet/prob1',                     #PNet face classification
                 'pnet/conv4-2/BiasAdd',           #PNet BoundingBox Regression
                 'rnet/prob1',                     #RNet face classification
                 'rnet/conv5-2/conv5-2', #RNet BoundingBox Regression 
                 'onet/prob1',                     #ONet face classification
                 'onet/conv6-2/conv6-2', #ONet BoundingBox Regression
                 'onet/conv6-3/conv6-3'  #ONet Facial Landmark
                 ]
    graphDef = convert_variables_to_constants(sess, sess.graph_def, output_node_names=output_name)
    with tf.gfile.GFile("mtcnn_freezed_model.pb", 'wb') as f:
       f.write(graphDef.SerializeToString())
       
if __name__=="__main__":
    print('Creating networks and loading parameters') 
    sess = tf.Session() 
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    print("[*]MTCNN load success")
    freeze(sess)
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor