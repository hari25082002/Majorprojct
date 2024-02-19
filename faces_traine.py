import os 
import cv2 as cv 
import numpy as np 

people = []

for i in os.listdir(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces'):
    people.append(i)

# print(people)
DIR = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces'

haarcascade_path = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Testing_codes\face_detect.xml'  
haarcascade = cv.CascadeClassifier(haarcascade_path)

features = []
labels = []  

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  

        for img in os.listdir(path):
            image_path = os.path.join(path, img)
            
            # Check if the image is successfully read
            img_array = cv.imread(image_path)
            if img_array is None:
                print(f"Error reading image: {image_path}")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            face = haarcascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in face:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)  

create_train()

print("done______________")

features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv.face_LBPHFaceRecognizer.create() 

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
