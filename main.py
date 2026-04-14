import asyncio
import base64
import io
import os
import pathlib
import secrets
import shutil
import threading
import wave
import requests
import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Qwen Voice Clone")

_basic_security = HTTPBasic()
_BASIC_USER = os.getenv("BASIC_AUTH_USER", "hollycrm")
_BASIC_PASS = os.getenv("BASIC_AUTH_PASS", "hollycrm")


def _verify_basic(credentials: HTTPBasicCredentials = Depends(_basic_security)):
    ok_user = secrets.compare_digest(credentials.username.encode(), _BASIC_USER.encode())
    ok_pass = secrets.compare_digest(credentials.password.encode(), _BASIC_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Basic"},
        )

API_KEY = os.getenv("DASHSCOPE_API_KEY")
CUSTOMIZATION_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
TTS_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
UPLOAD_DIR = pathlib.Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

dashscope.api_key = API_KEY

# API 固定返回 24000Hz PCM_24000HZ_MONO_16BIT
_SAMPLE_RATE_SRC = 24000


def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


# ===== WebSocket TTS 回调：收集 PCM 音频数据 =====

class _TTSCollector(QwenTtsRealtimeCallback):
    def __init__(self):
        self._done = threading.Event()
        self._chunks: list[bytes] = []
        self.error: str | None = None

    def on_open(self) -> None:
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        self._done.set()

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get("type", "")
            if event_type == "response.audio.delta":
                self._chunks.append(base64.b64decode(response["delta"]))
            elif event_type == "session.finished":
                self._done.set()
            elif event_type == "error":
                self.error = str(response)
                self._done.set()
        except Exception as e:
            self.error = str(e)
            self._done.set()

    def wait(self, timeout: float = 60.0) -> bool:
        return self._done.wait(timeout)

    def pcm_bytes(self) -> bytes:
        return b"".join(self._chunks)


def _resample(pcm: bytes, src_rate: int, tgt_rate: int) -> bytes:
    import numpy as np
    if src_rate == tgt_rate:
        return pcm
    data = np.frombuffer(pcm, dtype=np.int16)
    new_len = int(len(data) * tgt_rate / src_rate)
    resampled = np.interp(
        np.linspace(0, len(data) - 1, new_len),
        np.arange(len(data)),
        data,
    ).astype(np.int16)
    return resampled.tobytes()


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _run_tts_ws(text: str, voice: str, model: str, sample_rate: int, mode: str = "line") -> bytes:
    """同步执行 WebSocket TTS，在同一会话中逐行/逐字 append_text，返回 WAV 字节。"""
    if mode == "line":
        chunks = [l for l in text.split("\n") if l.strip()]
    else:  # char
        chunks = list(text)

    if not chunks:
        raise ValueError("文本内容为空")

    collector = _TTSCollector()
    client = QwenTtsRealtime(model=model, callback=collector, url=TTS_WS_URL)
    client.connect()
    client.update_session(
        voice=voice,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode="server_commit",
    )
    for chunk in chunks:
        client.append_text(chunk)
    client.finish()

    if not collector.wait(timeout=120):
        raise TimeoutError("TTS WebSocket 超时")
    if collector.error:
        raise RuntimeError(f"TTS 错误: {collector.error}")

    pcm = _resample(collector.pcm_bytes(), _SAMPLE_RATE_SRC, sample_rate)
    return _pcm_to_wav(pcm, sample_rate)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", dependencies=[Depends(_verify_basic)])
async def index():
    return FileResponse("static/qwen-voice-clone.html")


@app.get("/api/voices")
async def list_voices():
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {"action": "list", "page_size": 100, "page_index": 0},
    }
    resp = requests.post(CUSTOMIZATION_URL, json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"API 错误: {resp.text}")
    return resp.json()


class CreateVoiceRequest(BaseModel):
    preferred_name: str
    audio_data: str  # base64 data URI，如 "data:audio/wav;base64,..."
    target_model: str = "qwen3-tts-vc-realtime-2026-01-15"


@app.post("/api/voices")
async def create_voice(req: CreateVoiceRequest):
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": req.target_model,
            "preferred_name": req.preferred_name,
            "audio": {"data": req.audio_data},
        },
    }
    resp = requests.post(CUSTOMIZATION_URL, json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"创建音色失败: {resp.text}")
    return resp.json()


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {"action": "delete", "voice": voice_id},
    }
    resp = requests.post(CUSTOMIZATION_URL, json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"删除音色失败: {resp.text}")
    return resp.json()


class TTSRequest(BaseModel):
    text: str
    voice: str
    model: str = "qwen3-tts-vc-realtime-2026-01-15"
    sample_rate: int = 8000
    audio_format: str = "wav"
    mode: str = "line"  # "line"=逐行, "char"=逐字


@app.post("/api/tts")
async def tts(req: TTSRequest):
    try:
        loop = asyncio.get_event_loop()
        wav_data = await loop.run_in_executor(
            None, _run_tts_ws, req.text, req.voice, req.model, req.sample_rate, req.mode
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {e}")
    return Response(content=wav_data, media_type="audio/wav")


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    allowed = {".mp3", ".wav", ".m4a"}
    suffix = pathlib.Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="仅支持 MP3、WAV、M4A 格式")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}


@app.get("/api/uploads")
async def list_uploads():
    files = [
        {"filename": f.name, "size": f.stat().st_size}
        for f in UPLOAD_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".m4a"}
    ]
    return {"files": sorted(files, key=lambda x: x["filename"])}


@app.get("/api/uploads/{filename}")
async def get_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))
