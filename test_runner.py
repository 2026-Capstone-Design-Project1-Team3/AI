"""
test_runner.py
영상 파일을 입력받아 gesture / gaze / voice / fluency / report 모델을 모두 실행하고 결과를 출력합니다.

사용법:
    python test_runner.py --video 영상경로.mp4
    python test_runner.py --video 영상경로.mp4 --calib 캘리브레이션영상경로.mp4
    python test_runner.py --video 영상경로.mp4 --sway 0.05 --fidget 0.02 --gaze 0.05
    python test_runner.py --video 영상경로.mp4 --script 대본.txt --type 0
    python test_runner.py --video 영상경로.mp4 --skip voice fluency
"""

import argparse
import base64
import json
import os
import sys


def load_video_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_gesture(video_path: str, sway_thresh: float, fidget_thresh: float):
    print("\n" + "=" * 50)
    print("📹  GESTURE 분석")
    print("=" * 50)

    from gesture_model import GestureAnalyzer
    import numpy as np

    analyzer = GestureAnalyzer()
    data = analyzer.collect_landmarks(video_path)

    if not data["shoulder_mid_x"]:
        print("  ❌ 분석 데이터 없음 (얼굴/상반신이 잘 보이지 않을 수 있어요)")
        return

    sway_score   = float(np.std(data["shoulder_mid_x"]))
    fidget_score = float(np.mean(data["wrist_movement"])) if data["wrist_movement"] else 0.0

    feedbacks = []
    if sway_score > sway_thresh:
        feedbacks.append("좌우로 움직임이 많음")
    if fidget_score > fidget_thresh:
        feedbacks.append("손동작이 산만함")

    print(f"  sway  score  : {sway_score:.4f}  (threshold={sway_thresh})")
    print(f"  fidget score : {fidget_score:.4f}  (threshold={fidget_thresh})")
    print(f"  피드백        : {feedbacks if feedbacks else '이상 없음'}")


def run_gaze(video_path: str, calib_path: str | None, gaze_thresh: float):
    print("\n" + "=" * 50)
    print("👁️   GAZE 분석")
    print("=" * 50)

    from gaze_calibration import calculate_calibration_values
    from gaze_model import analyze_gaze_chunk, calculate_gaze_score, calculate_gaze_distribution
    import gaze_model

    if calib_path:
        print(f"  캘리브레이션 영상: {calib_path}")
        calib_b64 = load_video_as_base64(calib_path)
    else:
        print("  캘리브레이션 영상 없음 → 분석 영상으로 대체")
        calib_b64 = load_video_as_base64(video_path)

    calib = calculate_calibration_values(calib_b64)
    if calib is None:
        print("  ❌ 캘리브레이션 실패 (얼굴이 인식되지 않았어요)")
        return

    l_offset, r_offset, ratio = calib
    print(f"  캘리브레이션 결과 → left_offset={l_offset:.4f}, right_offset={r_offset:.4f}, ratio={ratio:.4f}")

    original_thresh = gaze_model.GAZE_THRESHOLD
    gaze_model.GAZE_THRESHOLD = gaze_thresh

    video_b64 = load_video_as_base64(video_path)
    history = analyze_gaze_chunk(video_b64, l_offset, r_offset)

    gaze_model.GAZE_THRESHOLD = original_thresh

    score = calculate_gaze_score(history)
    dist  = calculate_gaze_distribution(history)

    print(f"  GAZE_THRESHOLD : {gaze_thresh}")
    print(f"  분석 프레임 수  : {len(history)}")
    print(f"  카메라 응시율   : {dist['camera']}%")
    print(f"  화면  응시율    : {dist['screen']}%")
    print(f"  시선 점수       : {score} / 100")


def run_voice(video_path: str):
    print("\n" + "=" * 50)
    print("🎙️   VOICE 분석")
    print("=" * 50)

    from voice_model import analyse_voice_model

    video_b64 = load_video_as_base64(video_path)
    result = analyse_voice_model(video_b64)

    print(f"  전체 발화속도(SPM) : {result['overall_spm']}")
    print(f"  영상 길이          : {result['total_duration']}초")
    print(f"  인식된 전체 텍스트 : {result['full_text'][:80]}{'...' if len(result['full_text']) > 80 else ''}")

    print(f"\n  [구간별 속도]")
    for iv in result["interval_analysis"]:
        print(f"    {iv['range']} → {iv['spm']} SPM")

    if result["silence_log"]:
        print(f"\n  [침묵 구간]")
        for s in result["silence_log"]:
            print(f"    {s['at_second']}초 지점 → {s['duration']}초 침묵")
    else:
        print(f"\n  [침묵 구간] 없음")


def run_fluency(video_path: str):
    print("\n" + "=" * 50)
    print("🎤  FLUENCY 분석 (떨림 + 침묵)")
    print("=" * 50)

    from fluency_model import analyse_fluency_model

    video_b64 = load_video_as_base64(video_path)
    result = analyse_fluency_model(video_b64)

    t = result["tremor"]
    print("\n  [음성 떨림]")
    print(f"    레벨         : {t['level']}")
    print(f"    떨림 점수    : {t['tremor_score']} / 100  (높을수록 불안정)")
    print(f"    피치 표준편차: {t['pitch_std']} Hz")
    print(f"    에너지 변동  : {t['energy_cv']}")
    print(f"    Jitter       : {t['jitter']}")
    for fb in t["feedbacks"]:
        print(f"    → {fb}")

    s = result["silence"]
    print("\n  [침묵 패턴]")
    if s:
        for entry in s:
            print(f"    {entry['at_second']}초 지점 → {entry['duration']}초 침묵")
    else:
        print("    침묵 구간 없음")


def run_report(
    video_path : str,
    calib_path : str | None,
    script_path: str | None,
    analysis_type: int,
):
    print("\n" + "=" * 50)
    print("📊  REPORT 생성 (통합 분석)")
    print("=" * 50)

    from report_model import generate_report
    from gaze_calibration import calculate_calibration_values

    # 대본 로드
    script = ""
    if script_path:
        if not os.path.exists(script_path):
            print(f"  ❌ 대본 파일을 찾을 수 없어요: {script_path}")
            return
        with open(script_path, "r", encoding="utf-8") as f:
            script = f.read()
        print(f"  대본 파일: {script_path}")
    else:
        print("  대본 없음 → finalScore = 0")

    # 캘리브레이션
    calib_b64 = load_video_as_base64(calib_path if calib_path else video_path)
    calib = calculate_calibration_values(calib_b64)
    l_offset = r_offset = 0.0
    if calib:
        l_offset, r_offset, _ = calib
        print(f"  캘리브레이션 → left={l_offset:.4f}, right={r_offset:.4f}")
    else:
        print("  ❌ 캘리브레이션 실패 → offset 0.0 으로 대체")

    video_b64 = load_video_as_base64(video_path)

    result = generate_report(
        test_id       = "test-local-001",
        file_key      = os.path.basename(video_path),
        video_b64     = video_b64,
        script        = script,
        analysis_type = analysis_type,
        l_offset      = l_offset,
        r_offset      = r_offset,
    )

    # JSON 파일로 저장
    output_path = os.path.splitext(video_path)[0] + "_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ JSON 저장 완료: {output_path}")
    print(f"\n  [결과 미리보기]")
    print(f"    시선 점수       : {result['gazeScore']}")
    print(f"    시선 분포       : {result['gazeDistribution']}")
    print(f"    유창성 레벨     : {result['fluencyLevel']} (0:하 1:중 2:상)")
    print(f"    유창성 피드백   : {result['fluencyFeedback'][:50]}...")
    print(f"    속도 점수       : {result['speedScore']}")
    print(f"    속도 분포       : {result['speedDistribution']}")
    print(f"    제스처 키워드   : {result['gestureFeedbackWord']}")
    print(f"    제스처 문장     : {result['gestureFeedbackSentence']}")
    print(f"    최종 점수       : {result['finalScore']}")
    print(f"    발화 텍스트     : {result['transcript'][:60]}...")


def main():
    parser = argparse.ArgumentParser(description="AI 모델 테스트 러너")
    parser.add_argument("--video",   required=True,       help="분석할 영상 경로")
    parser.add_argument("--calib",   default=None,        help="캘리브레이션 영상 경로")
    parser.add_argument("--script",  default=None,        help="대본 텍스트 파일 경로 (.txt)")
    parser.add_argument("--type",    type=int, default=0, help="0:발표 / 1:면접 (기본 0)")
    parser.add_argument("--sway",    type=float, default=0.05, help="sway threshold (기본 0.05)")
    parser.add_argument("--fidget",  type=float, default=0.03, help="fidget threshold (기본 0.03)")
    parser.add_argument("--gaze",    type=float, default=0.05, help="gaze threshold (기본 0.05)")
    parser.add_argument("--skip",    nargs="*",  default=[],
                        choices=["gesture", "gaze", "voice", "fluency", "report"],
                        help="건너뛸 모듈 (예: --skip voice fluency)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 영상 파일을 찾을 수 없어요: {args.video}")
        sys.exit(1)

    print(f"\n🎬 영상: {args.video}")
    print(f"   sway={args.sway} / fidget={args.fidget} / gaze={args.gaze} / type={args.type}")

    if "gesture" not in args.skip:
        run_gesture(args.video, args.sway, args.fidget)

    if "gaze" not in args.skip:
        run_gaze(args.video, args.calib, args.gaze)

    if "voice" not in args.skip:
        run_voice(args.video)

    if "fluency" not in args.skip:
        run_fluency(args.video)

    if "report" not in args.skip:
        run_report(args.video, args.calib, args.script, args.type)

    print("\n" + "=" * 50)
    print("✅  분석 완료")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()