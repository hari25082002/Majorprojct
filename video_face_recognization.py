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
while True:
    is_true, frame = capture.read()
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = haascascade.detectMultiScale(gray, scaleFactor=2.5, minNeighbors=5)
    print(f'Number of faces: {len(faces)}')
    for (x, y, w, h) in faces:
        faces_roi = gray[y:y+h, x:x+w]
        # Resize the detected face for recognition
        faces_roi = cv.resize(faces_roi, (100, 100))
        # Recognize the face using the trained model
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'{people[label]} with confidence: {confidence}')
        # Display the recognized person's name and draw a rectangle around the face
        cv.putText(frame, str(people[label]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (242,242, 12), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)
    # Display the frame
    cv.imshow('frame', frame)
    # Break the loop if 's' key is pressed
    if cv.waitKey(3) & 0xFF == ord('s'):
        break
# Release the capture object and close all windows
capture.release()
cv.destroyAllWindows()
