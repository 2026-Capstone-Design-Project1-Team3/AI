from faster_whisper import WhisperModel
import os
import base64

model = WhisperModel("base", device="cpu", compute_type="int8")

def analyze_speed(video_base64):
    video_data = base64.b64decode(video_base64)
    with open("temp_chunk.webm", "wb") as f:
        f.write(video_data)

    
    segments, info = model.transcribe("temp_chunk.webm", language="ko")

    full_text = ""
    for segment in segments:
        full_text += segment.text

    char_count = len(full_text.replace(" ", ""))
    duration = 20

    cps = char_count / duration

    score = calculate_speed_socre(cps)

    return score



def calculate_speed_socre(cps):
    if 4.2 <= cps <= 4.6:
        return 0
    elif cps> 4.6:
        return 1
    elif cps < 4.2:
        return 2
