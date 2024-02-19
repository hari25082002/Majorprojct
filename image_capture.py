import cv2
import os

# Set the path to the directory where you want to save the images
save_path = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces\Hari path'

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object
cap = cv2.VideoCapture(0)  # Change 0 to the camera index if you want to capture from a camera

# Counter for the number of frames saved
frame_count = 0

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)

    # Save the frame as an image only if a face is detected
    if len(faces) > 0:
        image_path = os.path.join(save_path, f'frame_{frame_count}.png')
        cv2.imwrite(image_path, frame)
        frame_count += 1

    # Break the loop if 50 frames are saved
    if frame_count == 50:
        break

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
