import cv2 as cv 


img = cv.imread('1.jpg')


resize= cv.resize(img,(500,500),interpolation=cv.INTER_AREA)
#cv.imshow('Resized ',resize)

gray= cv.cvtColor(resize,cv.COLOR_BGR2GRAY)
#cv.imshow('GRAY ',gray)

haascascade= cv.CascadeClassifier('face_detect.xml')

face= haascascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

print(f'no of face {len(face)}')

for(x,y,w,h) in face:
    cv.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),thickness=3)
cv.imshow("Dected face",resize)










cv.waitKey(0)


