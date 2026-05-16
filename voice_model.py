from faster_whisper import WhisperModel
import os, base64, uuid
from moviepy import VideoFileClip


from realtime_voice import calculate_speed_score
model = WhisperModel("base", device = "cpu", compute_type="int8")

def analyze_silence(all_segments_data, threshold = 2.0): # 3초 이상 쉬면 어색하다고 느낌
    silence_logs = []

    for i in range(1, len(all_segments_data)):
        prev_end = all_segments_data[i-1]["end"]
        curr_start = all_segments_data[i]["start"]

        silence_gap = curr_start - prev_end

        if silence_gap > threshold:
            silence_logs.append({
                "at_second": round(prev_end, 2),
                "duration" : round(silence_gap, 2)
            })
    
    return silence_logs

def analyse_voice_model(video_data_base64, interval_seconds = 20):
    video_bytes = base64.b64decode(video_data_base64)
    temp_video_path = f"temp_{uuid.uuid4()}.webm"
    temp_audio_path = f"temp_{uuid.uuid4()}.wav"

    result = 0
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        video = VideoFileClip(temp_video_path)
        total_duration = video.duration
        video.audio.write_audiofile(temp_audio_path, codec ='pcm_s16le')

        segments, info = model.transcribe(temp_audio_path, beam_size=5) # 속도 정확도 사이 파라미터 조절..~~

        all_segments_data = []
        total_syllables = 0
        actual_duration = 0

        for segment in segments:
            text_clean = segment.text.strip()
            syllable_count = len(text_clean.replace(" ",""))

            all_segments_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": text_clean,
                "count": syllable_count
            })

            actual_duration += (segment.end - segment.start)

            total_syllables += syllable_count
        
        interver_speeds = []

        for start_time in range(0, int(total_duration), interval_seconds):
            end_time = start_time + interval_seconds
            interval_count = 0
            speaking_duration = 0

            for s in all_segments_data:
                if start_time <= s["start"] < end_time:
                    interval_count += s["count"]
                    speaking_duration += (s["end"] - s["start"])

            spm = (interval_count /speaking_duration)*60

            interver_speeds.append({
                "range": f"{start_time}s ~ {end_time}s",
                "spm": round(spm, 2)
            })

        total_spm = (total_syllables/actual_duration)*60

        return {
            "overall_spm": round(total_spm, 2),
            "total_duration" : round(total_duration, 2),
            "interval_analysis": interver_speeds,
            "full_text": ". ".join([s["text"] for s in all_segments_data]),
            "silence_log" : analyze_silence(all_segments_data)
        }


    finally:
        video.close()
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if os.path.exists(temp_audio_path) : os.remove(temp_audio_path)
