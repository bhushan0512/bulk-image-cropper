import cv2
import os
import numpy as np

# Define the paths for the source and destination folders
source_folder = 'images'
destination_folder = 'output'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error opening image {image_path}.")
        return

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, skip this image
    if len(faces) == 0:
        print(f"No face detected in {image_path}. Skipping...")
        return

    # Iterate over all detected faces
    for (x, y, w, h) in faces:
        # Expand the bounding box by 30% to ensure extra margin around the face
        margin = int(w * 0.3)  # 30% margin around the detected face
        startX = max(0, x - margin)
        startY = max(0, y - int(h * 0.5))  # Extra margin above the head
        endX = min(image.shape[1], x + w + margin)
        endY = min(image.shape[0], y + h + int(h * 0.5))  # Extra margin below the chin

        # Crop the face region with additional margin
        cropped = image[startY:endY, startX:endX]

        # Save the cropped image without resizing
        cv2.imwrite(output_path, cropped)
        print(f"Cropped image saved to {output_path}")
        return  # Process only the first detected face

# Process each image in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename.rsplit('.', 1)[0] + '.jpg')
        print(f"Processing {source_path}...")
        crop_face(source_path, destination_path)

print('Processing complete.')
