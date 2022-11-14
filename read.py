from multiprocessing.connection import wait
import cv2 as cv

#reading image
img = cv.imread('photos/download.jpg')
cv.imshow('download', img)
cv.waitKey(0)

#resize
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

#reading videos
capture = cv.VideoCapture('video\VID_20220412_181656.mp4')
while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
    cv.imshow('video Resized', frame_resized)

    if cv.waitKey(50) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()