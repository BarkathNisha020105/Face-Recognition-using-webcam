import cv2
import os

# Create directory for dataset if it doesn't exist
if not os.path.exists('face_dataset'):
    os.makedirs('face_dataset')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Ask for person's name
person_name = input("Enter the person's name: ").lower()
count = 0

# Create directory for this person
person_dir = os.path.join('face_dataset', person_name)
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

print("Capturing images. Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save face when 'c' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured face
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            img_name = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_name, face_img)
            print(f"Saved {img_name}")
            count += 1

    cv2.imshow('Create Dataset', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()