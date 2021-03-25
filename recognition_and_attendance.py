'''
        FACE RECOGNITION AND ATTENDANCE SYSTEM
'''

import cv2
import numpy as np
import csv
import datetime
import pickle

face_cascade = cv2.CascadeClassifier('/home/pratapkygo/Desktop/Face/haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

sub = input("Enter the subject ")

labels = {}
todays_date = datetime.date.today()
names, num, status = [], [], []

with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f) #(name:id)
	labels = {v:k for k,v in og_labels.items()} #(id:name)

attendance = {k:[v, 0] for k,v in og_labels.items()} #{name:[id, attendance_status]}

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

		if conf>=65:
			print(id_, labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			attendance[name][1] = 1
			color = (255,0,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

for x,y in attendance.items():
	names.append(x)
	num.append(y[0])
	if y[1] == 1:
		status.append('Present')
	else: status.append('Absent')

res = zip(names,num,status)

with open("Attendance_Report_"+"_"+sub+"_"+str(todays_date)+".csv", 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows([["Name","ID Number","Status"]])
    writer.writerows(res)

print("Attendance Report\n")

for k,v in attendance.items():
	if v[1]==1:
		print(k+" is present")
	else:
		print(k+" is absent")

cap.release()
cv2.destroyAllWindows()