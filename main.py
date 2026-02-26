import cv2
from deepface import DeepFace
import os

print("Starting DeepFace Recognition...")

known_image_path = "known_faces/akash.jpg"

if not os.path.exists(known_image_path):
    print("ERROR: akash.jpg not found")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    label = "No Face"

    try:
        result = DeepFace.verify(
            frame,
            known_image_path,
            model_name="Facenet",
            enforce_detection=False
        )

        # Add confidence threshold check
        if result["verified"] and result["distance"] < 0.6:
            label = "Akash"
        else:
            label = "Unknown"

    except:
        label = "No Face"

    # Draw label
    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("DeepFace Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()