'''
        MODEL TRAINER
'''


import os
import cv2
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		path = os.path.join(root, file)
		label = os.path.basename(root).lower()
		if label in label_ids:
			pass
		else:
			label_ids[label] = current_id
			current_id+=1

		id_ = label_ids[label]
		print(label_ids)

		img = cv2.imread(path,cv2. cv2.IMREAD_GRAYSCALE)
		image_array = np.array(img, "uint8")

		x_train.append(image_array)
		y_labels.append(id_)


#print(x_train)
#print(y_labels)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

print("Training Complete")