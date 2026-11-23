import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
# DeepFace's default threshold for FaceNet with L2-normalised Euclidean distance
THRESHOLD = 0.80


def normalize(embedding):
    vector = np.array(embedding)
    return vector / np.linalg.norm(vector)


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

        known_embeddings.append(normalize(embedding))
        known_names.append(name)

if not known_embeddings:
    raise SystemExit(f"No .jpg or .png images found in {KNOWN_FACES_DIR}/")

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

    results = DeepFace.extract_faces(frame, enforce_detection=False)
    frame_height, frame_width = frame.shape[:2]

    for result in results:
        area = result["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        # When no face is found, DeepFace returns the whole frame with
        # confidence 0 instead of raising, so skip that placeholder.
        if result.get("confidence", 0) == 0 and (w, h) == (frame_width, frame_height):
            continue

        face = frame[y:y+h, x:x+w]

        try:
            embedding = normalize(DeepFace.represent(
                img_path=face,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"])
        except Exception as error:
            print(f"Skipped a face: {error}")
            continue

        distances = [
            np.linalg.norm(embedding - known_embedding)
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

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()