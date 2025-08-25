import cv2
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')

# Load the id mapping
id_dict = np.load('label_ids.npy', allow_pickle=True).item()

# Reverse the dictionary to get name from id
id_to_name = {v: k for k, v in id_dict.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Font for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Recognize the face
        roi_gray = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(roi_gray)

        # If confidence is less than 70, consider it a match
        if confidence < 70:
            name = id_to_name.get(id, "Unknown")
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = ""

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name + confidence_text, (x, y - 10), font, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()