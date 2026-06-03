# CodeAlpha_Object_Detection_and_Tracking
Code Alpha AI/ML Internship Tasks 4 - Object Detection and Tracking


# Task 4: Object Detection and Tracking

Real-time object detection using YOLOv8 + IoU-based SORT tracker.

## Tech Stack
- Python, OpenCV, YOLOv8 (Ultralytics)

## How to Run
pip install ultralytics opencv-python numpy
python object_track_yolo.py          # webcam
python object_track_yolo.py --file video.mp4  # video file

## Features
- YOLOv8 detects 80 object classes in real time
- SORT-style tracker assigns persistent IDs
- Color-coded bounding boxes per tracked object
- Live FPS counter
