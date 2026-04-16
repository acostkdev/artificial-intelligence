import cv2 as cv
import numpy as np
import os

dataSet = 'dataset'
faces = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0
for face in faces:
    facePath = dataSet + '/' + face
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath + '/' + faceName, 0))
    label = label + 1

faceRecognizer = cv.face.LBPHFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('LBPHFace.xml')
print("Entrenamiento LBPH terminado")
