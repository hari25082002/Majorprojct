import numpy as np
import cv2 as cv
import os 

# Use absolute path for the cascade classifier file
haarcascade_path = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Testing_codes\face_detect.xml'  # Replace with the actual absolute path
haascascade = cv.CascadeClassifier(haarcascade_path)

# Load pre-trained features and labels
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')
people = []

for i in os.listdir(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces'):
    people.append(i)



# Create face recognizer and load the trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Open the webcam
capture = cv.VideoCapture(0)
# ... (previous code)

while True:
    is_true, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = haascascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    print(f'Number of faces: {len(faces)}')

    for (x, y, w, h) in faces:
        faces_roi = gray[y:y+h, x:x+w]
        faces_roi = cv.resize(faces_roi, (100, 100))

        label, confidence = face_recognizer.predict(faces_roi)

        # Set a confidence threshold (adjust the value as needed)
        if confidence < 100:
            name = people[label]
            print(f'{name} with confidence: {confidence}')

            cv.putText(frame, str(name), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

    cv.imshow('frame', frame)

    if cv.waitKey(3) & 0xFF == ord('s'):
        break

# ... (remaining code)
capture.release()
cv.destroyAllWindows()