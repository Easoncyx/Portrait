import numpy as np
import cv2 as cv




face_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt2.xml')
upperbody_cascade = cv.CascadeClassifier('../data/haarcascades/haarcascade_mcs_upperbody.xml')

window_size = 23
display_mode = 'every_frame' # default only object detected frame
# input_mode = 'image' # default video
input_mode = 'video' # default video

if input_mode == 'image':
    img = cv.imread('../img/8.jpg')
else:
    cap = cv.VideoCapture(0)
    # cap = cv.VideoCapture('../img/vtest.avi')


while(True):
    if input_mode != 'image':
        ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    body_flag = False
    face_flag = False

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
    elif len(faces) >= 2: # multiple faces
        face_flag = True
        tempImg = img.copy()
        maskShape = (img.shape[0], img.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        np.array(sorted(faces, key=lambda x: x[2], reverse=True))
        if faces[1][2] / faces[0][2] <= 0.7: # draw only first face
            (x,y,w,h) = faces[0]
            tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
            cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
            cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
        else: # draw all faces
            for (x, y, w, h) in faces:
                # tempImg = cv.blur(tempImg, (window_size, window_size))
                tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
                cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
                cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
        mask_inv = cv.bitwise_not(mask)
        img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
        img_fg = cv.bitwise_and(img, img, mask=mask)
        img = cv.add(img_bg, img_fg)
    else: # no face
        bodies = upperbody_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(30, 30))

        if len(bodies) >= 1:
            body_flag = True
            tempImg = img.copy()
            maskShape = (img.shape[0], img.shape[1], 1)
            mask = np.full(maskShape, 0, dtype=np.uint8)
            for (x, y, w, h) in bodies:
                # roi_gray = gray[y:y+h, x:x+w]
                # roi_color = img[y:y+h, x:x+w]
                # faces_in_body = face_cascade.detectMultiScale(roi_gray)
                # for (fx,fy,fw,fh) in faces_in_body:
                #     cv.circle(roi_gray, (int((fx + fx + fw) / 2), int((fy + fy + fh) / 2)), int(fh * 0.6), (0, 255, 0), 5)
                # tempImg = cv.blur(tempImg, (window_size, window_size))
                tempImg = cv.GaussianBlur(tempImg, (window_size, window_size), 0)
                cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 3)
                cv.rectangle(mask, (x,y), (x+w,y+h), (255), -1)
            mask_inv = cv.bitwise_not(mask)
            img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
            img_fg = cv.bitwise_and(img, img, mask=mask)
            img = cv.add(img_bg, img_fg)

    if display_mode == 'every_frame':
        cv.imshow('img',img)
    else:
        if body_flag or face_flag:
            cv.imshow('img',img)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
    if input_mode == 'image':
        cv.waitKey(0)
        break
if input_mode != 'image':
    cap.release()
cv.destroyAllWindows()
