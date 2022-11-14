import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    check, frame = capture.read() 
    assert check   
    cv.imshow('whiteboard', frame)
    Key = cv.waitKey(1)
    if (Key == ord('q')):
        break 

capture.release()
cv.destroyAllWindows()

 
