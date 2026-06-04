from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from gaze_calibration import calculate_calibration_values
from realtime_voice import analyze_speed
from report_model import generate_report
from pydantic import BaseModel
from typing import Optional
import asyncio, os, httpx, boto3, base64
from jose import jwt, JWTError

app = FastAPI()

SPRING_SERVER_URL = os.getenv("SPRING_SERVER_URL", "http://localhost:8080")
INTERNAL_SECRET   = os.getenv("INTERNAL_SECRET", "")
S3_BUCKET         = os.getenv("S3_BUCKET", "")
AWS_REGION        = os.getenv("AWS_REGION", "ap-south-1")

PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA5zQYcQlkX7tiETUEReTu
C0jawfwUd4GD8rkjs+KO2V+Ytv4bqA7y4OWZTpsnuHNIidBeWgCJqaC+NWAg2QVk
NY1FWzTuGYodAY6WqWiSpTf/PkJrvbtyv2nARS3iqkDEdrBfOCCNC5R9vTIoowHw
2dnVOBOYVSinHL2n0RFSjIrs1WPgP/RzixK4Ye75IMNJt+8yMdr5cLiwpQ6Pp91S
Tb6FLLNJWQE1DauL8QFqzQDKuCygJi9NqZF4z+VP8oboMplGbGiq20L2oshg8NG0
jIjYARh9nHEsfKClU3kW00FRTzn+S4SLIApF3Nbt+rxxGgXxkLSAm0sSEN/WGRnG
mwIDAQAB
-----END PUBLIC KEY-----"""


@app.websocket("/ws/analysis")
async def websocket_data(
    websocket     : WebSocket,
    folderId      : str,
    token         : str,
    leftEyeOffset : float = 0.0,
    rightEyeOffset: float = 0.0,
    ratio         : float = 0.0,
):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "VIDEO_CHUNK":
                score = analyze_speed(data["videoData"])

                await websocket.send_json({
                    "type"       : "SPEED_RESULT",
                    "currentTime": data["currentTime"],
                    "speedScore" : score,
                })

            elif data["type"] == "CALIBRATION_CHUNK":
                calib = calculate_calibration_values(data["videoData"])

                if calib is None:
                    await websocket.send_json({
                        "type"   : "CALIBRATION_DONE",
                        "success": False,
                    })
                    continue

                left_ratio, right_ratio, rat = calib

                asyncio.create_task(
                    _send_eye_calibration(
                        user_id      = get_user_id_from_token(token),
                        left_offset  = left_ratio,
                        right_offset = right_ratio,
                        ratio        = rat,
                    )
                )

                await websocket.send_json({
                    "type"          : "CALIBRATION_DONE",
                    "success"       : True,
                    "leftEyeOffset" : left_ratio,
                    "rightEyeOffset": right_ratio,
                    "ratio"         : rat,
                })

    except WebSocketDisconnect:
        pass


class EyeCalibration(BaseModel):
    leftEyeOffset : float = 0.0
    rightEyeOffset: float = 0.0
    ratio         : float = 0.0


class AnalysisRequest(BaseModel):
    analysisId     : str
    fileKey        : str
    type           : int
    extraInfo      : str = ""
    eyeCalibration : EyeCalibration = EyeCalibration()


@app.post("/analysis/start")
async def analysis_start(
    req              : AnalysisRequest,
    x_internal_secret: str = Header(...),
):
    if x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    asyncio.create_task(
        run_analysis(
            analysis_id   = req.analysisId,
            file_key      = req.fileKey,
            analysis_type = req.type,
            extra_info    = req.extraInfo,
            l_offset      = req.eyeCalibration.leftEyeOffset,
            r_offset      = req.eyeCalibration.rightEyeOffset,
        )
    )
    return {"status": 200}


async def run_analysis(
    analysis_id  : str,
    file_key     : str,
    analysis_type: int,
    extra_info   : str,
    l_offset     : float,
    r_offset     : float,
):
    try:
        video_b64 = await download_from_s3(file_key)
        script    = extra_info if analysis_type == 0 else ""

        result = generate_report(
            test_id       = analysis_id,
            file_key      = file_key,
            video_b64     = video_b64,
            script        = script,
            analysis_type = analysis_type,
            l_offset      = l_offset,
            r_offset      = r_offset,
        )
        await send_result_to_spring(result)

    except Exception as e:
        print(f"[분석 오류] {e}")


async def download_from_s3(file_key: str) -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response    = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
    video_bytes = response["Body"].read()
    return base64.b64encode(video_bytes).decode("utf-8")


async def send_result_to_spring(result: dict):
    headers = {"X-Internal-Secret": INTERNAL_SECRET}
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SPRING_SERVER_URL}/analysis",
            json    = result,
            headers = headers,
            timeout = 10,
        )


async def _send_eye_calibration(
    user_id     : str,
    left_offset : float,
    right_offset: float,
    ratio       : float,
):
    headers = {"X-Internal-Secret": INTERNAL_SECRET}
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{SPRING_SERVER_URL}/user/eye",
            json={
                "userId"        : user_id,
                "leftEyeOffset" : left_offset,
                "rightEyeOffset": right_offset,
                "ratio"         : ratio,
            },
            headers = headers,
            timeout = 10,
        )


def get_user_id_from_token(token: str) -> str:
    try:
        payload = jwt.decode(
            token,
            PUBLIC_KEY,
            algorithms=["RS256"],
        )
        return payload.get("sub", "")
    except JWTError:
        return ""