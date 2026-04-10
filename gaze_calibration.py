import cv2 as cv
import mediapipe as mp
import numpy as np
import base64

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5 # 해보고 수치 조절하기
)

def calculate_calibration_values(video_data):
    video_bytes = base64.b64decode(video_data)

    nparr = np.frombuffer(video_bytes, np.unit8)
    frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if frame is None: return None # 오류 코드 만들면 그걸로 쓰기

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return 0.0, 0.0, 0.0
    
    landmarks = results.multi_face_landmarks[0].landmark

    def get_eye(start_idx, end_idx, iris_idx):
        eye_left = landmarks[start_idx].x
        eye_right = landmarks[end_idx].x
        iris_x = landmarks[iris_idx].x

        ratio = (iris_x - eye_left) / (eye_right - eye_left)

        return ratio

    left_ratio = get_eye(33, 133, 468)
    right_ratio = get_eye(362, 263, 473)

    left_eye_offset = left_ratio - 0.5
    right_eye_offset = right_ratio -0.5

    eye_height = landmarks[159].y - landmarks[145].y
    eye_width = landmarks[133].x - landmarks[33].x

    ratio = eye_height/ eye_width

    return left_eye_offset, right_eye_offset, ratio