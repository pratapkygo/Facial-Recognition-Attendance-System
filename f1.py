import cv2
import numpy as np
import os
import pickle

face_cascade = cv2.CascadeClassifier('/home/pratapkygo/Desktop/Face/haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels = {}


with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f) #(name:id)
	labels = {v:k for k,v in og_labels.items()} #(id:name)

attendance = {k:0 for k,_ in og_labels.items()}
print(attendance)

recognizer.read("trainer.yml")

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		color = (100,150,200)
		stroke = 3
		cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

		id_,conf = recognizer.predict(roi_gray)
		print(conf)

		if conf>=75:
			print(id_, labels[id_])
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			attendance[name] = 1
			color = (255,0,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print("Attendance Report\n")

for k,v in attendance.items():
	if v==1:
		print(k+" is present")
	else:
		print(k+" is absent")


cap.release()
cv2.destroyAllWindows()