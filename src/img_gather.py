from __future__ import print_function
import argparse
import cv2 as cv


cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt2.xml')
upperbody_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_mcs_upperbody.xml')

# img = cv.imread('../img/1.jpeg')
i = 0
while(True):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
    bodies = upperbody_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(30, 30))
    if len(faces) >= 1 or len(bodies) >= 1:
        i += 1
        cv.imwrite("../img/img_" + str(i) + ".jpg", img)
        cv.imshow('img',img)
        if (cv.waitKey(500) & 0xFF == ord('q')) or i > 20:
            break
cap.release()
cv.destroyAllWindows()
