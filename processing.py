import os
import cv2
import numpy as np

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Standardize Lighting Conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    standardized_image = clahe.apply(equalized_image)
    
    return standardized_image

# Function to augment the image
def augment_image(image):
    # Rotation
    angle = np.random.randint(-30, 30)
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    
    # Scaling
    scale_factor = np.random.uniform(0.8, 1.2)
    scaled_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor)
    
    # Flipping (horizontal)
    if np.random.choice([True, False]):
        flipped_image = cv2.flip(scaled_image, 1)
    else:
        flipped_image = scaled_image
    
    return flipped_image

# Directory paths
input_directory = r'C:\Users\hsri2\Desktop\HTML _ css codes\majorproject\Faces\Hari'
output_directory = 'processed_images'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process and augment each image in the input directory
for i, filename in enumerate(os.listdir(input_directory)):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image_path = os.path.join(input_directory, filename)
        original_image = cv2.imread(image_path)
        
        # Preprocess the image
        processed_image = preprocess_image(original_image)
        
        # Augment the image
        augmented_image = augment_image(processed_image)
        
        # Save the processed and augmented images with numbered names
        output_filename = f'{i + 1}.jpg'
        output_path = os.path.join(output_directory, output_filename)
        cv2.imwrite(output_path, augmented_image)

print("Processing and augmentation complete. Images saved in the 'processed_images' directory.")
