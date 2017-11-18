import cv2
import numpy as np
import sys

#facePath = "/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
#smilePath = "/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier('/home/triplikehue/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml')
smileCascade = cv2.CascadeClassifier('/home/triplikehue/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('/home/triplikehue/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imSmile = cv2.imread('/home/triplikehue/Documentos/smileTest.png')
imAngry = cv2.imread('/home/triplikehue/Documentos/angryTest.png')
#cv2.namedWindow("output", cv2.WINDOW_GUI_NORMAL)

sF = 1.05
olhosCerrados = 0


while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.9,
            minNeighbors=22,
            minSize=(52, 52),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        # Set region of interest for smiles
        for (x, y, w, h) in smile:
            print "Sorriso", len(smile)
            cv2.imshow("Smile Detector", imSmile)
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

        eyes = eye_cascade.detectMultiScale(
            roi_color,
            scaleFactor=1.8,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        eyescerrados = eye_cascade.detectMultiScale(
            roi_color,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (ax, ay, aw, ah) in eyescerrados:
            print "Olhos cerrados", len(eyescerrados)
            if olhosCerrados == 5:
                cv2.imshow("Angry Detector", imAngry)
            cv2.rectangle(roi_color, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 2)

    #cv2.cv.Flip(frame, None, 1)
    cv2.imshow('Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or olhosCerrados == 5:
        break

cap.release()
cv2.destroyAllWindows()