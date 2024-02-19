import numpy as np
import cv2 as cv


haascascade= cv.CascadeClassifier('face_detect.xml')
features = np.load('features.npy', allow_pickle=True)

labels=np.load('labels.npy')
people = ['Hari', 'Dhoni','chandu']

face_recognizer = cv.face_LBPHFaceRecognizer.create() 

face_recognizer.read('face_trained.yml')

img=cv.imread(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Detect\Dhoni\2.jpeg')
#resize= cv.resize(img,(1000,1000),interpolation=cv.INTER_AREA)


gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#cv.imshow('gray',gray)

face= haascascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
   
for (x,y,w,h) in face:
    faces_roi=gray[y:y+h,x:x+h]

    label,confidence= face_recognizer.predict(faces_roi)
    print(f'{people[label]} with the confidence of{confidence}')

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

cv.imshow('deteced face ',img)

cv.waitKey(0)


