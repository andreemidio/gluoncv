import cv2
import numpy as np
import glob
 

width = 640
height =  480
size = (width,height)
FPS = 15

img_array = []
for filename in glob.glob('frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('projectPoseEstimation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()