import cv2

face_cascade = cv2.CascadeClassifier('/home/pratapkygo/Desktop/Face/haarcascades/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

count =0

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		color = (100,150,200)
		stroke = 3
		cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)
		count+=1

	if len(faces)==4:
		cv2.imwrite("Result.jpg",frame)

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()