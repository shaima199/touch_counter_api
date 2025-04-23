from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import shutil
import os
import gdown

app = FastAPI()

MODEL_PATH = "yolov8n.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1lJm6TFmcssok5xKAc-J44ymm0R7QhQy3"

# تحميل الموديل إذا ما كان موجود
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLOv8 model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("Download complete.")

model = YOLO(MODEL_PATH)
mp_pose = mp.solutions.pose

@app.post("/predict_touches")
async def predict_touches(video: UploadFile = File(...)):
    video_path = f"temp_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(video_path)
    touch_count = 0
    prev_touch = False
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)
        ball_detected = any(result.cls == 32 for result in results[0].boxes)  # class 32 = ball

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        foot_detected = False

        if pose_results.pose_landmarks:
            foot_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            if foot_landmark.visibility > 0.5:
                foot_detected = True

        if ball_detected and foot_detected:
            if not prev_touch:
                touch_count += 1
                prev_touch = True
        else:
            prev_touch = False

    cap.release()
    os.remove(video_path)
    return {"touches": touch_count}

