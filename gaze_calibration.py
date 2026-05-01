import cv2 as cv
import mediapipe as mp
import numpy as np
import base64
import uuid
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5 # 해보고 수치 조절하기
)

def get_eye(landmarks, start_idx, end_idx, iris_idx):
    eye_left = landmarks[start_idx].x
    eye_right = landmarks[end_idx].x
    iris_x = landmarks[iris_idx].x
    return (iris_x - eye_left) / (eye_right - eye_left)

def calculate_calibration_values(video_data, sample_count=10): # sample_count 조절 가능
    video_bytes = base64.b64decode(video_data)
    temp_path = f"temp_calib_{uuid.uuid4()}.webm"

    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv.VideoCapture(temp_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // sample_count)

        left_offsets = []
        right_offsets = []
        ratios = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark


                    left_ratio = get_eye(landmarks, 33, 133, 468)
                    right_ratio = get_eye(landmarks, 362, 263, 473)

                    left_offsets.append(left_ratio - 0.5)
                    right_offsets.append(right_ratio - 0.5)

                    eye_height = landmarks[159].y - landmarks[145].y
                    eye_width = landmarks[133].x - landmarks[33].x
                    if eye_width >0:
                        ratios.append(eye_height / eye_width)

            frame_idx += 1

        cap.release()

        if not left_offsets:
            return None

        return (
            sum(left_offsets) / len(left_offsets),
            sum(right_offsets) / len(right_offsets),
            sum(ratios) / len(ratios)
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)