import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

device = cv2.VideoCapture(0)

while True:
    ret, img = device.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_colour = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (0,0,255), 2)
        for (eex, eey, eew, eeh) in smile:
            cv2.rectangle(roi_colour, (eex, eey), (eex + eew, eey + eeh), (0, 255, 0), 2)
    cv2.imshow("face ID", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

device.release()
cv2.destroyAllWindows()
