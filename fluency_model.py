import os
import base64
import uuid

import numpy as np
import librosa
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

from voice_model import analyze_silence  # 침묵 분석 재사용

_whisper = WhisperModel("base", device="cpu", compute_type="int8")


# 임계값... 테스트 돌려보고 조절하기
TREMOR_PITCH_STD_THRESHOLD = 25.0   # Hz  : 피치 표준편차
TREMOR_ENERGY_CV_THRESHOLD = 0.55   # 무차원: 에너지 변동계수
TREMOR_JITTER_THRESHOLD    = 0.03   # 3%  : 프레임 간 피치 흔들림



def _extract_audio(video_base64: str) -> tuple[str, str, float]:
    uid = uuid.uuid4()
    temp_video = f"temp_fluency_video_{uid}.webm"
    temp_audio = f"temp_fluency_audio_{uid}.wav"

    video_bytes = base64.b64decode(video_base64)
    with open(temp_video, "wb") as f:
        f.write(video_bytes)

    clip = VideoFileClip(temp_video)
    total_duration = clip.duration
    clip.audio.write_audiofile(temp_audio, codec="pcm_s16le", logger=None)
    clip.close()

    return temp_video, temp_audio, total_duration


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass



def _compute_tremor(y: np.ndarray, sr: int) -> dict:

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]

    if len(voiced_f0) < 10:
        return {
            "pitch_std": 0.0, "energy_cv": 0.0, "jitter": 0.0,
            "tremor_score": 0, "level": "분석 불가",
            "feedbacks": ["유성음 구간이 너무 짧아 분석할 수 없습니다."],
        }

    pitch_std = float(np.std(voiced_f0))
    jitter    = float(np.mean(np.abs(np.diff(voiced_f0))) / np.mean(voiced_f0))

    rms       = librosa.feature.rms(y=y)[0]
    energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-8))

    score            = _tremor_score(pitch_std, energy_cv, jitter)
    level, feedbacks = _tremor_feedback(pitch_std, energy_cv, jitter)

    return {
        "pitch_std"   : round(pitch_std, 2),
        "energy_cv"   : round(energy_cv, 3),
        "jitter"      : round(jitter, 4),
        "tremor_score": score,
        "level"       : level,
        "feedbacks"   : feedbacks,
    }


def _tremor_score(pitch_std: float, energy_cv: float, jitter: float) -> int:
    p = min(pitch_std / (TREMOR_PITCH_STD_THRESHOLD * 2), 1.0)
    e = min(energy_cv / (TREMOR_ENERGY_CV_THRESHOLD * 2), 1.0)
    j = min(jitter    / (TREMOR_JITTER_THRESHOLD * 2),    1.0)
    return int(round((0.4 * p + 0.3 * e + 0.3 * j) * 100))


def _tremor_feedback(pitch_std, energy_cv, jitter):
    feedbacks = []

    if pitch_std > TREMOR_PITCH_STD_THRESHOLD:
        feedbacks.append(
            f"음 높낮이 변동이 큽니다 (std={pitch_std:.1f} Hz) "
        )
    if energy_cv > TREMOR_ENERGY_CV_THRESHOLD:
        feedbacks.append(
            "목소리 크기가 불규칙합니다 "
        )
    if jitter > TREMOR_JITTER_THRESHOLD:
        feedbacks.append(
            f"피치 흔들림(jitter={jitter*100:.1f}%)이 감지됩니다 "
        )

    count = sum([
        pitch_std  > TREMOR_PITCH_STD_THRESHOLD,
        energy_cv  > TREMOR_ENERGY_CV_THRESHOLD,
        jitter     > TREMOR_JITTER_THRESHOLD,
    ])
    level = "안정" if count == 0 else ("약간 불안정" if count == 1 else "불안정")

    if not feedbacks:
        feedbacks.append("음성이 안정적입니다.")

    return level, feedbacks



def analyse_fluency_model(video_base64: str) -> dict:
    temp_video, temp_audio, _ = _extract_audio(video_base64)

    try:
        # Whisper — 세그먼트 타임스탬프 추출
        segments_iter, _ = _whisper.transcribe(temp_audio, beam_size=5)
        all_segments_data = [
            {
                "start": s.start,
                "end"  : s.end,
                "text" : s.text.strip(),
                "count": len(s.text.strip().replace(" ", "")),
            }
            for s in segments_iter
        ]

        # librosa — 오디오 배열 로드
        y, sr = librosa.load(temp_audio, sr=None)

        tremor_result  = _compute_tremor(y, sr)
        silence_result = analyze_silence(all_segments_data)  # voice_model 함수 그대로 재사용

    finally:
        _cleanup(temp_video, temp_audio)

    return {
        "tremor" : tremor_result,
        "silence": silence_result,
    }