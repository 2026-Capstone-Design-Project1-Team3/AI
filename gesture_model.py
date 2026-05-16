import cv2 as cv
import mediapipe as mp
import numpy as np
import json

class GestureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            model_complexity = 1, # 사후분석이니까 정확도 생각... 성능 보고 ㄱㅊ다 싶으면 2로 바꾸기
            min_detection_confidence = 0.5, # 사람이라고 판단하는 정도
            min_tracking_confidence = 0.5 # 랜드마크 유지하는 확신도
        )
    
    def collect_landmarks(self, video_path):
        cap = cv.VideoCapture(video_path)
        data = {
            'shoulder_mid_x' : [],
            'wrist_movement' : [],
            'shoulder_y_diff' : []
        }

        prev_wrists = None
        try:
            while cap.isOpened():
                ret,frame = cap.read()
                if not ret: break

                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    mid_x = (lm[11].x + lm[12].x) /2
                    data['shoulder_mid_x'].append(mid_x)

                    y_diff = abs(lm[11].y - lm[12].y)
                    data['shoulder_y_diff'].append(y_diff)

                    curr_wrists = np.array([lm[15].x, lm[15].y, lm[16].x, lm[16].y])
                    if prev_wrists is not None:
                        dist = np.linalg.norm(curr_wrists - prev_wrists)
                        data['wrist_movement'].append(dist)
                
                    prev_wrists = curr_wrists
        finally:
            cap.release()
            
        return data
    
    def generate_report(self, data):
        sway_score = np.std(data['shoulder_mid_x'])

        fidget_score = np.mean(data['wrist_movement']) if data['wrist_movement'] else 0

        feedbacks = []
        if sway_score > 0.05: # 테스트 여러 개 돌려보고 임계값 조절하기
            feedbacks.append("좌우로 움직임이 많음")
        if fidget_score > 0.03:
            feedbacks.append("손동작이 산만함")
        
        return {
            # "sway_intensity" : round(float(sway_score), 2),
            # "fidget_intensity": round(float(fidget_score), 2),
            "feedbacks" : feedbacks,
        }