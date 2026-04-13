# AI Analysis Server

발표 및 면접 연습 중 발화 속도 분석, 연습 후 사용자의 **비언어적 요소(시선, 발화 속도, 몸짓)**를 분석하고 피드백 데이터를 생성하는 AI 엔진 서버

---

### 🛠 기술 스택 (Tech Stack)

| 분류 | 기술 |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Framework** | FastAPI (Asynchronous) |
| **Computer Vision** | MediaPipe (Iris/Pose), OpenCV |
| **Speech Analysis** | Faster-Whisper, Parselmouth |
| **Communication** | WebSocket (Real-time), Axios/HTTPX |
| **Environment** | conda/ venv|

---

### 🎯 주요 기능 (Key Features)

* **실시간 발화 속도 분석**: `faster-whisper` 기반 초당 음절 수(CPS) 실시간 모니터링 및 실시간 피드백 전송
* **시선 추적 및 캘리브레이션**: MediaPipe Iris를 통한 개인별 시선 보정 및 중앙 응시 여부 판별
* **비언어적 표현 분석**: 상반신 포즈 트래킹을 통한 제스처 적절성 및 신체 안정성 평가
* **정밀 리포트**: 연습 종료 후 전체 영상에 대한 시선 분포, 속도 추이, 유창성 지표 산출

---

### 📡 데이터 흐름 (Data Flow)

1. **Connection**: 프론트엔드와 **WebSocket** 연결 및 JWT 인증 수행
2. **Analysis**: 20초 단위 영상 조각(Chunk) 수신 → 시각/음성 지표 병렬 분석
3. **Feedback**: 실시간 분석 결과(`SPEED_RESULT`)를 클라이언트로 즉시 반환
4. **Reporting**: 연습 종료 후 최종 분석 데이터를 **Spring 백엔드** 서버로 전송 (`POST /analysis`)

---

### 🚀 시작하기 (Quick Start)

```bash
# 가상환경 활성화
conda activate AI

# 필수 라이브러리 설치
pip install -r 

# 서버 실행 (Uvicorn)
uvicorn main:app --host 0.0.0.0 --port 8000
