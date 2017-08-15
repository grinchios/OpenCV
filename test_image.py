from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

def facialRecognition():
    img = cv2.imread('out.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("Frame", img)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

camera = PiCamera()
camera.resolution = (320,240)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(320,240))

time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array

    cv2.imwrite('out.png',img)
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facialRecognition()
    
    #cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break
