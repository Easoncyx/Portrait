import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt2.xml')

# eye_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_eye.xml')


# img = cv.imread('../img/1.jpeg')

while(True):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = (cv.CASCADE_SCALE_IMAGE +
                     cv.CASCADE_DO_CANNY_PRUNING +
                     cv.CASCADE_FIND_BIGGEST_OBJECT +
                     cv.CASCADE_DO_ROUGH_SEARCH))
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv.imshow('img',img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
