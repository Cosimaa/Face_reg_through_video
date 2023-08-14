import os
import cv2 as cv
import numpy as np
import time

people = ['MARiA', 'Gong Linna']
DIR = r'E:\Projects\face recognition\Methods\HaarCascade_and_LBPH\Video\Datasets2'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)

            resize = cv.resize(img_array, (400, 400))                
            gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 10)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
start = time.time()
create_train()
end = time.time()
print('Training done ---------------')
print('training time: ' + str(end - start))


features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)
face_recognizer.save('face_trained2.yml')

np.save('features2.npy', features)
np.save('labels2.npy', labels)