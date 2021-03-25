import cv2
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/pratapkygo/Desktop/Face/haarcascades/haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)


	cv2.imshow('frame',frame)

	for x,y,w,h in faces:
		crop_img = frame[y:y+h, x:x+w]

	k = cv2.waitKey(10)&0xFF

	if k == ord('a'):
		gray2 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		name=input("Enter your name ")
		for i in range(1,51):
			i=str(i)
			filename=name+i+".jpg"
			color = (255,0,255)
			stroke = 2
			cv2.putText(frame, i, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			cv2.imwrite(filename, gray2)
			time.sleep(0.1)


	if k == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
