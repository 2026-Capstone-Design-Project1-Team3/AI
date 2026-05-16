"""
test_runner.py
영상 파일을 입력받아 gesture / gaze / voice / fluency 모델을 모두 실행하고 결과를 출력합니다.

사용법:
    python test_runner.py --video 영상경로.mp4
    python test_runner.py --video 영상경로.mp4 --calib 캘리브레이션영상경로.mp4
    python test_runner.py --video 영상경로.mp4 --sway 0.05 --fidget 0.02 --gaze 0.05
    python test_runner.py --video 영상경로.mp4 --skip voice fluency
"""

import argparse
import base64
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

    # 떨림
    t = result["tremor"]
    print("\n  [음성 떨림]")
    print(f"    레벨         : {t['level']}")
    print(f"    떨림 점수    : {t['tremor_score']} / 100  (높을수록 불안정)")
    print(f"    피치 표준편차: {t['pitch_std']} Hz")
    print(f"    에너지 변동  : {t['energy_cv']}")
    print(f"    Jitter       : {t['jitter']}")
    for fb in t["feedbacks"]:
        print(f"    → {fb}")

    # 침묵
    s = result["silence"]
    print("\n  [침묵 패턴]")
    if s:
        for entry in s:
            print(f"    {entry['at_second']}초 지점 → {entry['duration']}초 침묵")
    else:
        print("    침묵 구간 없음")


def main():
    parser = argparse.ArgumentParser(description="AI 모델 테스트 러너")
    parser.add_argument("--video",  required=True,      help="분석할 영상 경로")
    parser.add_argument("--calib",  default=None,       help="캘리브레이션 영상 경로 (없으면 분석 영상으로 대체)")
    parser.add_argument("--sway",   type=float, default=0.05, help="sway threshold (기본 0.05)")
    parser.add_argument("--fidget", type=float, default=0.03, help="fidget threshold (기본 0.03)")
    parser.add_argument("--gaze",   type=float, default=0.05, help="gaze threshold (기본 0.05)")
    parser.add_argument("--skip",   nargs="*",  default=[],
                        choices=["gesture", "gaze", "voice", "fluency"],
                        help="건너뛸 모듈 (예: --skip voice fluency)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 영상 파일을 찾을 수 없어요: {args.video}")
        sys.exit(1)

    print(f"\n🎬 영상: {args.video}")
    print(f"   sway={args.sway} / fidget={args.fidget} / gaze={args.gaze}")

    if "gesture" not in args.skip:
        run_gesture(args.video, args.sway, args.fidget)

    if "gaze" not in args.skip:
        run_gaze(args.video, args.calib, args.gaze)

    if "voice" not in args.skip:
        run_voice(args.video)

    if "fluency" not in args.skip:
        run_fluency(args.video)

    print("\n" + "=" * 50)
    print("✅  분석 완료")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()