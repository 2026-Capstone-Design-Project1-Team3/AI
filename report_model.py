from voice_model    import analyse_voice_model
from fluency_model  import _compute_tremor, _extract_audio, _cleanup
from gesture_model  import GestureAnalyzer
from gaze_model     import analyze_gaze_chunk, calculate_gaze_score, calculate_gaze_distribution
from script_model   import analyse_script_model
from voice_model    import analyze_silence

import base64, uuid, os
import librosa


# ── SPM 기준값 ────────────────────────────────────────────────────
SPM_SLOW_MAX = 4.2 * 60
SPM_FAST_MIN = 4.6 * 60

SCORE_OPTIMAL = 100
SCORE_FAST    = 50
SCORE_SLOW    = 30


def _save_temp_video(video_b64: str) -> str:
    temp_path = f"temp_report_{uuid.uuid4()}.webm"
    with open(temp_path, "wb") as f:
        f.write(base64.b64decode(video_b64))
    return temp_path


def _cleanup_path(path: str):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _calc_speed_distribution(interval_analysis: list) -> dict:
    if not interval_analysis:
        return {"fast": 0, "optimal": 0, "slow": 0}

    fast = optimal = slow = 0
    for iv in interval_analysis:
        spm = iv["spm"]
        if spm >= SPM_FAST_MIN:
            fast += 1
        elif spm <= SPM_SLOW_MAX:
            slow += 1
        else:
            optimal += 1

    total = len(interval_analysis)
    return {
        "fast"   : round(fast    / total * 100),
        "optimal": round(optimal / total * 100),
        "slow"   : round(slow    / total * 100),
    }


def _calc_speed_score(interval_analysis: list) -> int:
    if not interval_analysis:
        return 0

    scores = []
    for iv in interval_analysis:
        spm = iv["spm"]
        if SPM_SLOW_MAX < spm < SPM_FAST_MIN:
            scores.append(SCORE_OPTIMAL)
        elif spm >= SPM_FAST_MIN:
            scores.append(SCORE_FAST)
        else:
            scores.append(SCORE_SLOW)

    return round(sum(scores) / len(scores))


def _fluency_level_to_int(level: str) -> int:
    return {"안정": 2, "약간 불안정": 1, "불안정": 0}.get(level, 1)


def _gesture_to_feedback(feedbacks: list) -> tuple[str, str]:
    if not feedbacks:
        return "안정적", "제스처가 안정적입니다."

    keywords = []
    if any("흔들림" in fb for fb in feedbacks):
        keywords.append("좌우흔들림")
    if any("손동작" in fb for fb in feedbacks):
        keywords.append("손동작산만")

    keyword_str  = ", ".join(keywords) if keywords else "개선필요"
    sentence_str = " ".join(feedbacks)
    return keyword_str, sentence_str


def _gaze_to_feedback(gaze_score: int) -> str:
    if gaze_score >= 80:
        return "카메라를 잘 응시하고 있습니다."
    elif gaze_score >= 50:
        return "카메라 응시가 조금 부족합니다."
    else:
        return "카메라를 더 자주 응시해주세요."


def generate_report(
    test_id       : str,
    file_key      : str,
    video_b64     : str,
    script        : str,
    analysis_type : int,
    l_offset      : float,
    r_offset      : float,
) -> dict:

    temp_path = _save_temp_video(video_b64)
    temp_video_audio = None
    temp_audio = None

    try:
        # ── 1. 오디오 추출 (1회) ──────────────────────────────────
        temp_video_audio, temp_audio, _ = _extract_audio(video_b64)

        # ── 2. 음성 분석 (Whisper 1회) ────────────────────────────
        voice_result = analyse_voice_model(video_b64)
        full_text    = voice_result["full_text"]
        overall_spm  = voice_result["overall_spm"]

        # ── 3. 유창성 분석 (librosa, Whisper 재사용) ──────────────
        y, sr = librosa.load(temp_audio, sr=None)
        tremor = _compute_tremor(y, sr)
        silence_result = analyze_silence(voice_result.get("all_segments_data", []))

        # ── 4. 제스처 분석 ────────────────────────────────────────
        analyzer       = GestureAnalyzer()
        gesture_data   = analyzer.collect_landmarks(temp_path)
        gesture_report = analyzer.generate_report(gesture_data)
        gesture_kw, gesture_sentence = _gesture_to_feedback(gesture_report["feedbacks"])

        # ── 5. 시선 분석 ──────────────────────────────────────────
        gaze_history  = analyze_gaze_chunk(video_b64, l_offset, r_offset)
        gaze_score    = calculate_gaze_score(gaze_history)
        gaze_dist     = calculate_gaze_distribution(gaze_history)
        gaze_feedback = _gaze_to_feedback(gaze_score)

        # ── 6. 대본 유사도 분석 ───────────────────────────────────
        script_result = analyse_script_model(script, full_text)
        final_score   = script_result["similarity_score"]

        # ── 7. 속도 분포 / 점수 ──────────────────────────────────
        speed_dist  = _calc_speed_distribution(voice_result["interval_analysis"])
        speed_score = _calc_speed_score(voice_result["interval_analysis"])

        # ── 8. 유창성 레벨 → int ─────────────────────────────────
        fluency_level    = _fluency_level_to_int(tremor["level"])
        fluency_feedback = " ".join(tremor["feedbacks"])

    finally:
        _cleanup_path(temp_path)
        _cleanup(temp_video_audio, temp_audio)

    return {
        "analysisId"             : test_id,
        "gazeScore"              : gaze_score,
        "gazeFeedback"           : gaze_feedback,
        "gazeDistribution"       : gaze_dist,
        "fluencyLevel"           : fluency_level,
        "fluencyFeedback"        : fluency_feedback,
        "speedScore"             : speed_score,
        "speedSpm"               : overall_spm,
        "speedDistribution"      : speed_dist,
        "gestureFeedbackWord"    : gesture_kw,
        "gestureFeedbackSentence": gesture_sentence,
        "finalScore"             : final_score,
        "transcript"             : full_text,
        "fileKey"                : file_key,
        "type"                   : analysis_type,
    }