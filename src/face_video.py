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

    if len(faces) >= 1:
        tempImg = img.copy()
        maskShape = (img.shape[0], img.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)
        window_size = 23
        for (x, y, w, h) in faces:
            tempImg[:, :] = cv.blur(tempImg[:, :], (window_size, window_size))
            cv.circle(img, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (0, 255, 0), 5)
            cv.circle(mask, (int((x + x + w) / 2), int((y + y + h) / 2)), int(h * 0.6), (255), -1)
        mask_inv = cv.bitwise_not(mask)
        img_bg = cv.bitwise_and(tempImg, tempImg, mask=mask_inv)
        img_fg = cv.bitwise_and(img, img, mask=mask)
        dst = cv.add(img_bg, img_fg)

        cv.imshow('img',dst)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
