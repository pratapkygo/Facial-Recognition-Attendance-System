import cv2
import numpy as np
import csv
import datetime
import pickle

face_cascade = cv2.CascadeClassifier('/home/pratapkygo/Desktop/Face/haarcascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels = {}

recognizer.read("trainer.yml")

img = cv2.imread('1.jpeg')

font = cv2.FONT_HERSHEY_SIMPLEX

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
cv2.imshow('gray',gray)

while True:
	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
		Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

		if Id == 0:
			Id = "Mohith {0:.2f}%".format(round(100 - confidence, 2))
		elif Id == 1 :
			Id = "Surya {0:.2f}%".format(round(100 - confidence, 2))
		elif Id == 2:
			Id = "Pratap {0:.2f}%".format(round(100 - confidence, 2))
		elif Id == 3:
			Id = "Vaishnav {0:.2f}%".format(round(100 - confidence, 2))
		else:
			pass

		cv2.rectangle(img, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
		cv2.putText(img, str(Id), (x,y-40), font, 1, (255,255,255), 3)

	cv2.imshow('im',img) 

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cv2.imshow('res',img) 

cv2.waitKey(0)