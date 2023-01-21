import cv2 as cv 
import os
import numpy as np
import PIL
import math



def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)   


piece = []
background = []
images = os.path.join(r"C:\Users\OPE\Documents\Visual Studio 2008\computer vision\thepics.jpg")
img = cv.imread(images)
img_resize = rescaleFrame(img, scale = .15)
npimage = np.array(img_resize)


for i in range (img_resize.shape[0]):
    for j in range(img_resize.shape[1]):
        numbers = img_resize[i][j]
        if any(numbers < 197):
            background.append(numbers)
           #numbers[0], numbers[1], numbers[2] = [0, 0, 0]
        if all(numbers > 197):
            piece.append(numbers)
            #numbers[0], numbers[1], numbers[2] = [255, 255, 255]

#for values in piece:
    #for values in npimage:
        


cv.imshow('download', img_resize)
cv.waitKey(0) 
cv.destroyAllWindows()


