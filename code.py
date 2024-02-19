import cv2 as cv
import os
import numpy as np
import time
from datetime import datetime

class FACE_RECOGNITION:
    def __init__(self):
        
        self.capture = cv.VideoCapture(0)
        if not self.capture.isOpened():
           print("Error: Unable to open camera.")
           return
        self.haarcascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            is_true, frame = self.capture.read()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            faces = self.haarcascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
            
            if len(faces) > 0:
                self.Recognization(frame, gray, faces)
            else:
                self.Not_Recognization(frame)

            key = cv.waitKey(1)
            if key & 0xFF == ord('c'):
                self.capture.release()
                cv.destroyAllWindows()
                self.Create_face_dataset()

            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)
            
            cv.imshow('frame', frame)

            if key & 0xFF == ord('d'):
                break

        self.capture.release()
        cv.destroyAllWindows()
        
    def Recognization(self, frame, gray, faces):
        # print('Yes')
        haarcascade_path = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Testing_codes\face_detect.xml'  # Replace with the actual absolute path
        haascascade = cv.CascadeClassifier(haarcascade_path)
        
        # Load pre-trained features and labels
        features = np.load('features.npy', allow_pickle=True)
        labels = np.load('labels.npy')
        people = []

        for i in os.listdir(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces'):
            people.append(i)

        if not people:
            print("No people recognized.")
            return
        
        face_recognizer = cv.face.LBPHFaceRecognizer_create()
        face_recognizer.read('face_trained.yml')
        threshold=200
        for (x, y, w, h) in faces:
            faces_roi = gray[y:y+h, x:x+w]
            faces_roi = cv.resize(faces_roi, (100, 100))
        
            label, confidence = face_recognizer.predict(faces_roi)
            if confidence > threshold:

              print(f'{people[label]} with confidence: {confidence}')

              cv.putText(frame, str(people[label]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (242,242, 12), thickness=2)
              cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)
            else:
                self.Not_Recognization(frame)

        cv.imshow('frame', frame)
        
    def Not_Recognization(self, frame):
        # print('no')
        # print(confidence)
        path=r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Not detected faces'
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        not_recognize = os.path.join(path, f'not_detected_{current_datetime}.jpg')
        # gray_covert=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        # equalized_image = cv.equalizeHist(gray_covert)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # standardized_image = clahe.apply(equalized_image)

        cv.imwrite(not_recognize,frame)
        
    def Create_face_dataset(self):
        while True:
            try:
                print("Enter the Name of the person Which is saved as the file name")
                Name = input("").strip()
                if not Name:
                    print("Name cannot be empty. Please enter a valid name.")
                elif not Name.isalpha():
                    print("Name should only contain letters. Please enter a valid name.")
                else:
                    folder_path = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces\{}'.format(Name)
                    if os.path.exists(folder_path):
                        print("Folder with the same name already exists. Please enter a different name.")
                    else:
                        os.makedirs(folder_path)
                        print("Folder created successfully.")
                        self.Taking_Images(folder_path)
                        break
            except FileExistsError as e:
                print(e)
                
    def Taking_Images(self, folder_path):
        save_path = folder_path
        os.makedirs(save_path, exist_ok=True)
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
       
        cap = cv.VideoCapture(0) 
        if not cap.isOpened():
            print("Error: Unable to open camera.")
            return

        frame_count = 0
        try:
            while frame_count <= 3:
                ret, frame =cap.read()
                if not ret:
                    print("Error: unable to read frame from camera. ")
                    break
                try:
                    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
                except cv.error as e:
                    print(f"An error occurred while converting the frame to grayscale: {e}")
                    continue  
               
                faces=face_cascade.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

                if len(faces) > 0:
                    original_image_path = os.path.join(save_path, f'original_{frame_count}.png')
                    cv.imwrite(original_image_path, frame)
                    preprocessed_image = self.preprocess_image(gray_frame)
                    preprocessed_image_path = os.path.join(save_path, f'preprocessed_frame_{frame_count}.png')
                    cv.imwrite(preprocessed_image_path, preprocessed_image)
                    augmented_image = self.augment_image(gray_frame)
                    augmented_image_path = os.path.join(save_path, f'augmented_frame_{frame_count}.png')
                    cv.imwrite(augmented_image_path, augmented_image)
                   
                    frame_count += 1
                
                cv.imshow('Frame',frame)
                if cv.waitKey(1) & 0XFF==ord('q'):
                    break
                time.sleep(0.5)

        except Exception as e:
            print(f"An error occured :{e}")
        finally:
            cap.release()
            cv.destroyAllWindows()
            self.Traing_model()
        
    def preprocess_image(self, frame):
        if len(frame.shape) == 3:  
            gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:  
            gray_image = frame
        else:
            raise ValueError("Unsupported number of channels in input image")
        
        equalized_image = cv.equalizeHist(gray_image)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        standardized_image = clahe.apply(equalized_image)
        return standardized_image
    
    def augment_image(self, frame):
        angle = np.random.randint(-30, 30)
        rows, cols = frame.shape
        rotation_matrix = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv.warpAffine(frame, rotation_matrix, (cols, rows))
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_image = cv.resize(rotated_image, None, fx=scale_factor, fy=scale_factor)
        if np.random.choice([True, False]):
            flipped_image = cv.flip(scaled_image, 1)
        else:
            flipped_image = scaled_image
        return flipped_image
    
    def Traing_model(self):
        people = []

        for i in os.listdir(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces'):
            people.append(i)

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
        # print("Overwriting existing files...")

        face_recognizer.save('face_trained.yml')
        # print("Saved trained model to face_trained.yml")

        np.save('features.npy', features)
        # print("Saved features to features.npy")
        np.save('labels.npy', labels)
        # print("Saved labels to labels.npy")

        

    

              

# Example usage:
face_recognition = FACE_RECOGNITION()

