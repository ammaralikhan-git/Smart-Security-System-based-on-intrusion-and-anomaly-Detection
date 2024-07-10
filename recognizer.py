import cv2
import numpy as np
import face_recognition
import pickle

# Load known face encodings and names from a pickle file
with open('facedetection/src/face_encodings.pkl', 'rb') as file:
    encodings_dict = pickle.load(file)

known_encodings = []
known_names = []

# Iterate through the dictionary to populate lists
for name, encodings in encodings_dict.items():
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# Setup Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise Exception("Failed to load Haar Cascade classifier.")

def recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    recognized_names = []
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_face_image, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_face_image, face_locations)

        name = "Unknown"
        color = (0, 0, 255)  # Red color for unknown

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                color = (0, 255, 0)  # Green color for known

        recognized_names.append(name)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    return recognized_names,frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = recognize_faces(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
