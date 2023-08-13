import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['MARiA', 'Gong Linna']
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained2.yml')

cap = cv.VideoCapture(r"D:\Facebook.mp4")

while True:
    ret, img = cap.read()
    if ret:
        resize = cv.resize(img, (840, 720))
        gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    
        # Detect the face in the image
        faces_rect = haar_cascade.detectMultiScale(gray, 1.2, 10)

        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h+10,x:x+w+10]

            label, confidence = face_recognizer.predict(faces_roi)
            print(f'Label = {people[label]} with a confidence of {confidence}')

            cv.rectangle(resize, (x-20, y-20), (x+w + 20, y+h + 20), (255, 0, 0), 2) 
            cv.rectangle(resize, (x - 20, y + h - 20), (x + w + 20, y + h + 20), (255, 0, 0), cv.FILLED)
            cv.putText(resize, people[label], (x-20, y +h + 15), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Detected Face', resize)

        if cv.waitKey(5) == ord('q'):
            break
    else:
        break 
    
cap.release()
cv.destroyAllWindows()