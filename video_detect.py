import cv2 as cv

caputer= cv.VideoCapture(0)
haascascade= cv.CascadeClassifier('face_detect.xml')

while True:

    istrue,frame= caputer.read()
    
    face= haascascade.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=4)

    print(f'no of face {len(face)}')

    for(x,y,w,h) in face:
         cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)
    #cv.imshow("Dected face",frame)

    cv.imshow('frame',frame)
    
    if cv.waitKey(3)& 0XFF==ord('s'):
         break
    cv.imwrite(r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces\chandu',frame)
caputer.release()
cv.destroyAllWindows()


