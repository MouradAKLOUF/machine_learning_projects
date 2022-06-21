# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
video="ball_tracking_example.mp4"
buffer=64

# =============================================================================
# greenUpper = (200, 60, 60)
# greenLower = (100 , 10, 10)
# =============================================================================
redLower0 =(0, 70, 50)
redUpper0 =(10, 255, 255)

redLower1 =(170, 70, 50)
redUpper1 =(180, 255, 255)


pts = deque(maxlen=buffer)

# if a video path was not supplied, grab the reference
# to the webcam
vs = VideoStream(src=0).start()
#vs = cv2.VideoCapture(video)

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    frame = vs.read()
    frame = frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, redLower0, redUpper0)
    #mask = cv2.inRange(hsv, redLower1, redUpper1)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    pts.appendleft(center)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
vs.stop()
#vs.release()
# close all windows
cv2.destroyAllWindows()




# =============================================================================
# image = cv2.imread("WIN_20191006_16_06_35_Pro.jpg")
# (h, w, d) = image.shape
# blurred = cv2.GaussianBlur(image, (11, 11), 0)
# hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
# (h, w, d) = hsv.shape
# 
# cv2.imshow("Image", hsv)
# cv2.waitKey(0)
# 
# hsv[563,393,:]
# 
# hsv[563,555,:]
# 
# hsv[538,198,:]
# 
# hsv[718,380,:]
# 
# hsv[368,381,:]
# =============================================================================
