
"""
사용법:
    from report_model import generate_report

    result = generate_report(
        test_id   = "abc123",
        file_key  = "s3/videos/abc123.webm",
        video_b64 = video_base64,
        script    = "백엔드에서 받은 대본 텍스트",  # type=0(발표)일 때
        analysis_type = 0,  # 0: 발표, 1: 면접
        l_offset  = 0.0,
        r_offset  = 0.0,
    )
"""

import json
from voice_model    import analyse_voice_model
from fluency_model  import analyse_fluency_model
from gesture_model  import GestureAnalyzer
from gaze_model     import analyze_gaze_chunk, calculate_gaze_score, calculate_gaze_distribution
from script_model   import analyse_script_model

import base64, uuid, os
from moviepy import VideoFileClip


SPM_SLOW_MAX    = 4.2 * 60   # 252 SPM 이하 → slow
SPM_FAST_MIN    = 4.6 * 60   # 276 SPM 이상 → fast


def _save_temp_video(video_b64: str) -> str:
    temp_path = f"temp_report_{uuid.uuid4()}.webm"
    with open(temp_path, "wb") as f:
        f.write(base64.b64decode(video_b64))
    return temp_path


def _cleanup(path: str):
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
    """optimal 비율을 0~100 점수로 변환"""
    dist = _calc_speed_distribution(interval_analysis)
    return dist["optimal"]


def _fluency_level_to_int(level: str) -> int:
    """안정→2(상), 약간 불안정→1(중), 불안정→0(하)"""
    return {"안정": 2, "약간 불안정": 1, "불안정": 0}.get(level, 1)


def _gesture_to_feedback(feedbacks: list) -> tuple[str, str]:
    """
    gesture feedbacks → (키워드, 문장)
    피드백 없으면 긍정 문구 반환
    """
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


def generate_report(
    test_id       : str,
    file_key      : str,
    video_b64     : str,
    script        : str,
    analysis_type : int,   # 0: 발표, 1: 면접
    l_offset      : float,
    r_offset      : float,
) -> dict:
    """
    모든 분석 모델을 실행하고 백엔드 스펙 JSON 반환

    Parameters
    ----------
    test_id       : 연습기록 예비 ID (백엔드에서 수신)
    file_key      : S3 파일 키 (백엔드에서 수신)
    video_b64     : base64 인코딩된 전체 영상
    script        : 대본 텍스트 (백엔드에서 수신)
    analysis_type : 0=발표, 1=면접
    l_offset      : 시선 캘리브레이션 left offset
    r_offset      : 시선 캘리브레이션 right offset
    """

    temp_path = _save_temp_video(video_b64)

    try:
        # ── 1. 음성 분석 (Whisper 1회) ────────────────────────────
        voice_result = analyse_voice_model(video_b64)
        full_text    = voice_result["full_text"]

        # ── 2. 유창성 + 떨림 분석 (Whisper 재사용) ───────────────
        fluency_result = analyse_fluency_model(video_b64)
        tremor         = fluency_result["tremor"]

        # ── 3. 제스처 분석 ────────────────────────────────────────
        analyzer         = GestureAnalyzer()
        gesture_data     = analyzer.collect_landmarks(temp_path)
        gesture_report   = analyzer.generate_report(gesture_data)
        gesture_kw, gesture_sentence = _gesture_to_feedback(gesture_report["feedbacks"])

        # ── 4. 시선 분석 ──────────────────────────────────────────
        gaze_history  = analyze_gaze_chunk(video_b64, l_offset, r_offset)
        gaze_score    = calculate_gaze_score(gaze_history)
        gaze_dist     = calculate_gaze_distribution(gaze_history)

        # ── 5. 대본 유사도 분석 ───────────────────────────────────
        script_result = analyse_script_model(script, full_text)
        final_score   = script_result["similarity_score"]

        # ── 6. 속도 분포 / 점수 ──────────────────────────────────
        speed_dist  = _calc_speed_distribution(voice_result["interval_analysis"])
        speed_score = _calc_speed_score(voice_result["interval_analysis"])

        # ── 7. 유창성 레벨 → int ─────────────────────────────────
        fluency_level    = _fluency_level_to_int(tremor["level"])
        fluency_feedback = " ".join(tremor["feedbacks"])

    finally:
        _cleanup(temp_path)

    return {
        "testId"                 : test_id,
        "gazeScore"              : gaze_score,
        "gazeDistribution"       : gaze_dist,
        "fluencyLevel"           : fluency_level,
        "fluencyFeedback"        : fluency_feedback,
        "speedScore"             : speed_score,
        "speedDistribution"      : speed_dist,
        "gestureFeedbackWord"    : gesture_kw,
        "gestureFeedbackSentence": gesture_sentence,
        "finalScore"             : final_score,
        "transcript"             : full_text,
        "fileKey"                : file_key,
        "type"                   : analysis_type,
    }