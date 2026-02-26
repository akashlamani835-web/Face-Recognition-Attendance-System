import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
THRESHOLD = 0.6

if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

print("Loading known faces...")

known_embeddings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.endswith(".jpg") or file.endswith(".png"):
        path = os.path.join(KNOWN_FACES_DIR, file)
        name = os.path.splitext(file)[0]

        embedding = DeepFace.represent(
            img_path=path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        known_embeddings.append(embedding)
        known_names.append(name)

print("Faces loaded successfully.")

today_date = datetime.now().strftime("%Y-%m-%d")
csv_file = os.path.join(ATTENDANCE_DIR, f"{today_date}.csv")

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(csv_file, index=False)

def mark_attendance(name):
    df = pd.read_csv(csv_file)

    if name not in df["Name"].values:
        now = datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame([[name, now]], columns=["Name", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"Attendance marked for {name}")

cap = cv2.VideoCapture(0)

print("Starting camera... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.extract_faces(frame, enforce_detection=False)

        for result in results:
            x, y, w, h = result["facial_area"].values()
            face = frame[y:y+h, x:x+w]

            embedding = DeepFace.represent(
                img_path=face,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            distances = [
                np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                for known_embedding in known_embeddings
            ]

            min_distance = min(distances)
            index = distances.index(min_distance)

            if min_distance < THRESHOLD:
                name = known_names[index]
                label = name
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except:
        pass

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()