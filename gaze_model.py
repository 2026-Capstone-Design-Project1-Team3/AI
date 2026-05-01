import cv2 as cv
import numpy as np
import base64, uuid, os

from gaze_calibration import face_mesh, get_eye


GAZE_THRESHOLD = 0.05

def analyze_gaze_chunk(video_data_base64, l_offset, r_offset, sample_interval = None): #일단 interval none으로 해두고 CPU 부하 생각해서 샘플링 interval 조절
    video_bytes = base64.b64decode(video_data_base64)
    temp_path = f"temp_gaze_{uuid.uuid4()}.webm"
    results_list = []

    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        
        cap = cv.VideoCapture(temp_path)
        if not cap.isOpened():
            return results_list
        
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            if sample_interval is None or frame_idx % sample_interval == 0:
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_frame)

                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark

                                    
                    left_ratio = get_eye(landmarks, 33, 133, 468)
                    right_ratio = get_eye(landmarks, 362, 263, 473)

                    current_left_offset = left_ratio - 0.5
                    current_right_offset = right_ratio - 0.5

                    left_diff = abs(current_left_offset - l_offset)
                    right_diff = abs(current_right_offset - r_offset)

                    avg_diff = (left_diff + right_diff) / 2

                    if avg_diff < GAZE_THRESHOLD:
                        results_list.append("camera")
                    else:
                        results_list.append("screen")
            frame_idx += 1
        
        cap.release()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return results_list


def calculate_gaze_score(gaze_history):
    if not gaze_history:
        return 0
    
    camera_count = gaze_history.count("camera")

    return int((camera_count / len(gaze_history)) * 100)

def calculate_gaze_distribution(gaze_history):
    if not gaze_history:
        return {"screen" : 0, "camera" : 0}
    
    total = len(gaze_history)
    camera_count = gaze_history.count("camera")
    screen_count = total - camera_count

    return {
        "screen" : round((screen_count / total) * 100),
        "camera" : round((camera_count / total) * 100)
    }

