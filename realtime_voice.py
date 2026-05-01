from faster_whisper import WhisperModel
import os, base64, uuid

model = WhisperModel("base", device="cpu", compute_type="int8")

def analyze_speed(video_base64):
    video_bytes = base64.b64decode(video_base64)
    temp_path = f"temp_chunk_{uuid.uuid4()}.webm" # 여러 명이 동시 접속하면 AI 서버 내에 저장되는 이 데이터가 혼동될 수 있기 때문에 램덤 ID를 붙여서 파일명 다르게 해주기
    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

    
        segments, info = model.transcribe(temp_path, language="ko")

        full_text = ""
        for segment in segments:
            full_text += segment.text

        char_count = len(full_text.replace(" ", ""))
        duration = 20

        cps = char_count / duration

        score = calculate_speed_score(cps)

        return score
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



def calculate_speed_score(cps):
    if 4.2 <= cps <= 4.6:
        return 0
    elif cps> 4.6:
        return 1
    else:
        return 2
