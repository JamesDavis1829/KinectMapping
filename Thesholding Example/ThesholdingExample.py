import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    frame = cv2.blur(frame,(5,5))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
##    # Blue settings
##    lower_blue = np.array([90,120,40])
##    upper_blue = np.array([120,255,255])

    # Green settings
    lower_blue = np.array([40,60,40])
    upper_blue = np.array([100,255,255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    mask = cv2.bitwise_not(mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5)
    if k == 27:
        break

cv2.destroyAllWindows()
