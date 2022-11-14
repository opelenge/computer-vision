import cv2 as cv


def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)