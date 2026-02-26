# Face Recognition Attendance System

A deep learning based real-time attendance system using FaceNet embeddings.

## Features
- Real-time face recognition via webcam
- Multi-person support
- Automatic CSV attendance logging
- One entry per person per day
- Threshold-based similarity matching

## Tech Stack
- Python 3.10
- OpenCV
- DeepFace (FaceNet)
- TensorFlow
- Pandas

## How It Works
The system extracts 128-dimensional embeddings from faces using a pretrained FaceNet model.
It compares embeddings using Euclidean distance to identify individuals.
If similarity is within threshold, attendance is recorded in a daily CSV file.

## Future Improvements
- Database integration (MySQL)
- Web dashboard using Flask
- Multi-camera support
