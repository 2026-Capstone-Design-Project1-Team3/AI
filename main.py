from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gaze_calibration import calculate_calibration_values
from realtime_voice import analyze_speed
app = FastAPI()

@app.websocket("/ws/analysis")

async def websocket_data(
    websocket: WebSocket,
    forderId: str,
    leftEyeOffset: float,
    rightEyeOffet: float,
    ratio: float
):
    await websocket.accept()
    # 연결 실패시.. ~~ 구현
    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "VIDEO_CHUNK":
                score = analyze_speed(data["VideoData"])

                await websocket.send_json({
                    "type": "SPEED_RESULT",
                    "currentTime": data["currentTime"],
                    "speedScore": score
                })


            if data["type"] == "CALIBRATION_CHUNK":
                left_ratio, right_ratio, rat = calculate_calibration_values(data["VideoData"])

                # 백으로 보내주는 코드 짜기

                await websocket.send_json({
                    "type": "CALIBRAION_DONE"
                    # 프론트에도 값 넘겨줘야하는지 아닌지... 물어보기
                })

                
    except WebSocketDisconnect:
        # 마무리 로직.. 뭐 해야하지