import cv2
import numpy as np
import os
from PIL import Image

# Path to dataset
dataset_path = 'face_dataset'

# Create face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Function to get images and labels
def get_images_and_labels(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('jpg') or file.endswith('png'):
                image_paths.append(os.path.join(root, file))

    face_samples = []
    ids = []
    id_dict = {}
    current_id = 0

    for image_path in image_paths:
        # Get the label (person name) from the directory name
        person_name = os.path.basename(os.path.dirname(image_path))

        if person_name not in id_dict:
            id_dict[person_name] = current_id
            current_id += 1

        # Load image and convert to grayscale
        pil_image = Image.open(image_path).convert('L')
        image_np = np.array(pil_image, 'uint8')

        # Get the face from the image (assuming images are already faces)
        face_samples.append(image_np)
        ids.append(id_dict[person_name])

    return face_samples, ids, id_dict


print("Training face recognition model...")
faces, ids, id_dict = get_images_and_labels(dataset_path)

# Train the model
recognizer.train(faces, np.array(ids))

# Save the model
recognizer.save('face_recognizer.yml')

# Save the id mapping
np.save('label_ids.npy', id_dict)

print(f"Training completed. {len(faces)} images trained.")
print("ID mapping:", id_dict)