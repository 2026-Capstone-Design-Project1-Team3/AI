from faster_whisper import WhisperModel
import os, base64, uuid, subprocess, gc

from realtime_voice import calculate_speed_score
model = WhisperModel("base", device="cpu", compute_type="int8")


def _extract_audio_ffmpeg(video_path: str, audio_path: str):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-y", audio_path
    ], check=True, capture_output=True)


def _get_duration(video_path: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ], capture_output=True, text=True)
    
    duration_str = result.stdout.strip()
    if duration_str and duration_str != 'N/A':
        return float(duration_str)
    
    result2 = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "stream=duration",
        "-of", "csv=p=0",
        video_path
    ], capture_output=True, text=True)
    
    for line in result2.stdout.strip().split('\n'):
        line = line.strip()
        if line and line != 'N/A':
            return float(line)
    
    return 0.0


def analyze_silence(all_segments_data, threshold=2.0):
    silence_logs = []
    for i in range(1, len(all_segments_data)):
        prev_end    = all_segments_data[i-1]["end"]
        curr_start  = all_segments_data[i]["start"]
        silence_gap = curr_start - prev_end
        if silence_gap > threshold:
            silence_logs.append({
                "at_second": round(prev_end, 2),
                "duration" : round(silence_gap, 2)
            })
    return silence_logs


def analyse_voice_model_from_path(video_path: str, interval_seconds=20):
    temp_audio_path = f"temp_audio_{uuid.uuid4()}.wav"

    try:
        total_duration = _get_duration(video_path)
        _extract_audio_ffmpeg(video_path, temp_audio_path)

        segments, info = model.transcribe(
            temp_audio_path,
            beam_size=5,
            chunk_length=30,
            condition_on_previous_text=False
        )

        all_segments_data = []
        total_syllables   = 0
        actual_duration   = 0

        for segment in segments:
            text_clean     = segment.text.strip()
            syllable_count = len(text_clean.replace(" ", ""))
            all_segments_data.append({
                "start": segment.start,
                "end"  : segment.end,
                "text" : text_clean,
                "count": syllable_count
            })
            actual_duration += (segment.end - segment.start)
            total_syllables += syllable_count

        gc.collect()

        interval_speeds = []
        for start_time in range(0, int(total_duration), interval_seconds):
            end_time          = start_time + interval_seconds
            interval_count    = 0
            speaking_duration = 0
            for s in all_segments_data:
                if start_time <= s["start"] < end_time:
                    interval_count    += s["count"]
                    speaking_duration += (s["end"] - s["start"])
            spm = (interval_count / speaking_duration) * 60 if speaking_duration > 0 else 0
            interval_speeds.append({
                "range": f"{start_time}s ~ {end_time}s",
                "spm"  : round(spm, 2)
            })

        total_spm = (total_syllables / actual_duration) * 60 if actual_duration > 0 else 0

        return {
            "overall_spm"      : round(total_spm, 2),
            "total_duration"   : round(total_duration, 2),
            "interval_analysis": interval_speeds,
            "full_text"        : ". ".join([s["text"] for s in all_segments_data]),
            "silence_log"      : analyze_silence(all_segments_data),
            "all_segments_data": all_segments_data,
            "temp_audio_path"  : temp_audio_path,  # 다음 모델에서 재사용
        }

    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        raise e


def analyse_voice_model(video_data_base64, interval_seconds=20):
    video_bytes     = base64.b64decode(video_data_base64)
    temp_video_path = f"temp_{uuid.uuid4()}.webm"

    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)
        result = analyse_voice_model_from_path(temp_video_path, interval_seconds)
        return result
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)