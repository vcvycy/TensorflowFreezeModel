import sys
import argparse
import cv2
from scipy import misc
parser=argparse.ArgumentParser() 
parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare')
    
print(parser.parse_args(sys.argv[1:]))
#img=cv2.imread("1.jpg")
img=misc.imread("1.jpg",mode="RGB")
cv2.namedWindow("hello")
cv2.imshow("hello",img)
cv2.waitKey(0)