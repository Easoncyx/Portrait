import numpy as np
import cv2 as cv

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('../img/vtest.avi')
# face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt2.xml')

body_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_fullbody.xml')

window_size = 23
img = cv.imread('../img/2.jpg')

while(True):
    # ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body_flag = False
    face_flag = False
    bodies = body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15))

    if len(bodies) >= 1:
        body_flag = True
        tempImg = img.copy()
        maskShape = (img.shape[0], img.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        window_size = 23
        for (x, y, w, h) in bodies:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            faces_in_body = face_cascade.detectMultiScale(roi_gray)
            for (fx,fy,fw,fh) in faces_in_body:
                cv.circle(roi_gray, (int((fx + fx + fw) / 2), int((fy + fy + fh) / 2)), int(fh * 0.6), (0, 255, 0), 5)
            # tempImg = cv.blur(tempImg, (window_size, window_size))
            tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
            cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 3)
            cv.rectangle(mask, (x,y), (x+w,y+h), (255), -1)
        mask_inv = cv.bitwise_not(mask)
        img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
        img_fg = cv.bitwise_and(img, img, mask=mask)
        img = cv.add(img_bg, img_fg)
    else:
        faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))
        if len(faces) == 1: # if face is detacted
            face_flag = True
            tempImg = img.copy()
            maskShape = (img.shape[0], img.shape[1], 1)
            mask = np.full(maskShape, 0, dtype=np.uint8)
            (x, y, w, h) = faces[0]
            # tempImg = cv.blur(tempImg, (window_size, window_size))
            tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
            cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
            cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
            mask_inv = cv.bitwise_not(mask)
            img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
            img_fg = cv.bitwise_and(img, img, mask=mask)
            img = cv.add(img_bg, img_fg)
        elif len(faces) >= 2:
            face_flag = True
            tempImg = img.copy()
            maskShape = (img.shape[0], img.shape[1], 1)
            mask = np.full(maskShape, 0, dtype=np.uint8)
            np.array(sorted(faces, key=lambda x: x[2], reverse=True))
            if faces[0][2] - faces[1][2] >= faces[0][2]/2:
                (x,y,w,h) = faces[0]
                tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
                cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
                cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
            else:
                for (x, y, w, h) in faces:
                    # tempImg = cv.blur(tempImg, (window_size, window_size))
                    tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
                    cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
                    cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
            mask_inv = cv.bitwise_not(mask)
            img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
            img_fg = cv.bitwise_and(img, img, mask=mask)
            img = cv.add(img_bg, img_fg)
    if body_flag or face_flag:
        cv.imshow('img',img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
