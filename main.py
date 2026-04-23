import asyncio
import base64
import io
import json
import os
import re
import time
import numpy as np
import pathlib
import secrets
import shutil
import tempfile
import threading
import wave
import soxr
import yaml
import requests
import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from dashscope.audio.tts_v2 import (
    VoiceEnrollmentService,
    SpeechSynthesizer as CosySpeechSynthesizer,
    AudioFormat as CosyAudioFormat,
    ResultCallback,
)
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rustfs import upload_file as rustfs_upload

app = FastAPI(title="Qwen Voice Clone")

_basic_security = HTTPBasic()
_BASIC_USER = os.getenv("BASIC_AUTH_USER", "hollycrm")
_BASIC_PASS = os.getenv("BASIC_AUTH_PASS", "hollycrm")


def _verify_basic(credentials: HTTPBasicCredentials = Depends(_basic_security)):
    ok_user = secrets.compare_digest(
        credentials.username.encode(), _BASIC_USER.encode())
    ok_pass = secrets.compare_digest(
        credentials.password.encode(), _BASIC_PASS.encode())
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

# ===== CosyVoice 常量 =====
COSYVOICE_CUSTOMIZATION_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
COSYVOICE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
COSYVOICE_HTTP_URL = "https://dashscope.aliyuncs.com/api/v1"
COSYVOICE_REGISTRY_FILE = pathlib.Path("cosyvoice_registry.json")


def _load_cosyvoice_registry() -> list:
    if COSYVOICE_REGISTRY_FILE.exists():
        try:
            return json.loads(COSYVOICE_REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_cosyvoice_registry(registry: list) -> None:
    COSYVOICE_REGISTRY_FILE.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class _CosyVoiceCollector(ResultCallback):
    def __init__(self):
        self._done = threading.Event()
        self._chunks: list[bytes] = []
        self.error: str | None = None

    def on_open(self) -> None:
        pass

    def on_complete(self) -> None:
        self._done.set()

    def on_error(self, message: str) -> None:
        self.error = str(message)
        self._done.set()

    def on_close(self) -> None:
        self._done.set()

    def on_event(self, message) -> None:
        pass

    def on_data(self, data: bytes) -> None:
        self._chunks.append(data)

    def wait(self, timeout: float = 120.0) -> bool:
        return self._done.wait(timeout)

    def pcm_bytes(self) -> bytes:
        return b"".join(self._chunks)


def _run_cosyvoice_tts(text: str, voice: str, model: str, sample_rate: int, mode: str = "line", speech_rate: float = 1.0, pitch_rate: float = 1.0) -> bytes:
    if mode == "line":
        chunks = [ln for ln in text.split("\n") if ln.strip()]
    else:
        chunks = list(text)
    if not chunks:
        raise ValueError("文本内容为空")

    dashscope.base_websocket_api_url = COSYVOICE_WS_URL
    dashscope.base_http_api_url = COSYVOICE_HTTP_URL

    collector = _CosyVoiceCollector()
    if sample_rate == 8000:
        format = CosyAudioFormat.PCM_8000HZ_MONO_16BIT
    elif sample_rate == 16000:
        format = CosyAudioFormat.PCM_16000HZ_MONO_16BIT
    elif sample_rate == 24000:
        format = CosyAudioFormat.PCM_24000HZ_MONO_16BIT
    else:
        format = CosyAudioFormat.PCM_8000HZ_MONO_16BIT
    synthesizer = CosySpeechSynthesizer(
        model=model,
        voice=voice,
        format=format,
        callback=collector,
        speech_rate=speech_rate,
        pitch_rate=pitch_rate
    )
    for chunk in chunks:
        synthesizer.streaming_call(chunk)
    synthesizer.streaming_complete()

    if not collector.wait(timeout=120):
        raise TimeoutError("CosyVoice TTS 超时")
    if collector.error:
        raise RuntimeError(f"TTS 错误: {collector.error}")

    src_pcm = collector.pcm_bytes()

    return _pcm_to_wav(src_pcm, sample_rate)


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


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _run_tts_ws(text: str, voice: str, model: str, sample_rate: int, mode: str = "line", speech_rate: float = 1.0, pitch_rate: float = 1.0) -> bytes:
    """同步执行 WebSocket TTS，在同一会话中逐行/逐字 append_text，返回 WAV 字节。"""
    if mode == "line":
        chunks = [l for l in text.split("\n") if l.strip()]
    else:  # char
        chunks = list(text)
    print(chunks)
    if not chunks:
        raise ValueError("文本内容为空")

    collector = _TTSCollector()
    client = QwenTtsRealtime(model=model, callback=collector, url=TTS_WS_URL)
    client.connect()
    client.update_session(
        voice=voice,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        speech_rate=speech_rate,
        pitch_rate=pitch_rate,
        mode="server_commit",
    )
    for chunk in chunks:
        client.append_text(chunk)
    client.finish()

    if not collector.wait(timeout=120):
        raise TimeoutError("TTS WebSocket 超时")
    if collector.error:
        raise RuntimeError(f"TTS 错误: {collector.error}")
    # 使用 soxr.resample 进行重采样
    src_pcm = collector.pcm_bytes()
    if _SAMPLE_RATE_SRC == sample_rate:
        pcm = src_pcm
    else:
        src_np = np.frombuffer(src_pcm, dtype=np.int16)
        if len(src_np) == 0:
            pcm = b""
        else:
            # float32 for soxr, mono
            src_audio = src_np.astype(np.float32) / 32768.0
            resampled = soxr.resample(src_audio, _SAMPLE_RATE_SRC, sample_rate)
            resampled_int16 = np.clip(
                resampled * 32768.0, -32768, 32767).astype(np.int16)
            pcm = resampled_int16.tobytes()
    return _pcm_to_wav(pcm, sample_rate)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", dependencies=[Depends(_verify_basic)])
async def index():
    return FileResponse("static/index.html")


@app.get("/qwen", dependencies=[Depends(_verify_basic)])
async def qwen_index():
    return FileResponse("static/qwen-voice-clone.html")


@app.get("/api/voices")
async def list_voices():
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {"action": "list", "page_size": 100, "page_index": 0},
    }
    resp = requests.post(CUSTOMIZATION_URL, json=payload,
                         headers=get_headers())
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
    resp = requests.post(CUSTOMIZATION_URL, json=payload,
                         headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"创建音色失败: {resp.text}")
    return resp.json()


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {"action": "delete", "voice": voice_id},
    }
    resp = requests.post(CUSTOMIZATION_URL, json=payload,
                         headers=get_headers())
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
    speech_rate: float = 1.0
    pitch_rate: float = 1.0


@app.post("/api/tts")
async def tts(req: TTSRequest):
    try:
        loop = asyncio.get_event_loop()
        wav_data = await loop.run_in_executor(
            None, _run_tts_ws, req.text, req.voice, req.model, req.sample_rate, req.mode, req.speech_rate, req.pitch_rate
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


_DEFAULT_TEXTS_FILE = pathlib.Path("default_texts.yml")


@app.get("/api/default-texts")
async def get_default_texts():
    if not _DEFAULT_TEXTS_FILE.exists():
        return []
    with open(_DEFAULT_TEXTS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or []


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


# ===== CosyVoice Routes =====

@app.get("/cosyvoice", dependencies=[Depends(_verify_basic)])
async def cosyvoice_index():
    return FileResponse("static/cosyvoice-clone.html")


@app.get("/api/cosyvoice/voices")
async def cosyvoice_list_voices():
    payload = {
        "model": "voice-enrollment",
        "input": {"action": "list_voice", "page_size": 1000, "page_index": 0},
    }
    resp = requests.post(COSYVOICE_CUSTOMIZATION_URL, json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"API 错误: {resp.text}")
    data = resp.json()
    voice_list = data.get("output", {}).get("voice_list", [])
    voice_list.sort(key=lambda v: v.get("gmt_modified", ""), reverse=True)
    return {"voices": voice_list}


class CosyCreateVoiceRequest(BaseModel):
    voice_name: str
    audio_data: str  # base64 data URI: "data:audio/wav;base64,..."
    target_model: str = "cosyvoice-v3.5-plus"


@app.post("/api/cosyvoice/voices")
async def cosyvoice_create_voice(req: CosyCreateVoiceRequest):
    if not re.match(r"^[a-z0-9]{1,10}$", req.voice_name):
        raise HTTPException(
            status_code=400, detail="音色名称只能包含小写字母和数字，最多 10 个字符")

    # Decode base64 audio → save to temp WAV file in uploads/
    try:
        header, b64 = req.audio_data.split(",", 1)
        audio_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="音频数据格式错误")

    tmp_filename = f"{req.voice_name}_{int(time.time())}.wav"
    tmp_path = UPLOAD_DIR / tmp_filename
    tmp_path.write_bytes(audio_bytes)

    # Upload to rustfs to get a public URL
    try:
        public_url = rustfs_upload(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传到公网失败: {e}")

    # Create voice enrollment via DashScope
    dashscope.base_http_api_url = COSYVOICE_HTTP_URL
    dashscope.base_websocket_api_url = COSYVOICE_WS_URL

    service = VoiceEnrollmentService()
    try:
        voice_id = service.create_voice(
            target_model=req.target_model,
            prefix=req.voice_name,
            url=public_url,
            max_prompt_audio_length=30,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建音色失败: {e}")

    # Poll for status (up to ~60 s)
    status = "DEPLOYING"
    for _ in range(12):
        await asyncio.sleep(5)
        try:
            info = service.query_voice(voice_id=voice_id)
            status = info.get("status", "DEPLOYING")
            if status in ("OK", "UNDEPLOYED"):
                break
        except Exception:
            pass

    registry = _load_cosyvoice_registry()
    registry.append({
        "voice_id": voice_id,
        "display_name": req.voice_name,
        "target_model": req.target_model,
        "status": status,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_cosyvoice_registry(registry)
    return {"voice_id": voice_id, "status": status}


@app.get("/api/cosyvoice/voices/{voice_id:path}/status")
async def cosyvoice_voice_status(voice_id: str):
    dashscope.base_http_api_url = COSYVOICE_HTTP_URL
    service = VoiceEnrollmentService()
    try:
        info = service.query_voice(voice_id=voice_id)
        status = info.get("status", "DEPLOYING")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Update registry
    registry = _load_cosyvoice_registry()
    for v in registry:
        if v["voice_id"] == voice_id:
            v["status"] = status
            break
    _save_cosyvoice_registry(registry)
    return {"voice_id": voice_id, "status": status}


@app.delete("/api/cosyvoice/voices/{voice_id:path}")
async def cosyvoice_delete_voice(voice_id: str):
    payload = {
        "model": "voice-enrollment",
        "input": {"action": "delete_voice", "voice_id": voice_id},
    }
    resp = requests.post(COSYVOICE_CUSTOMIZATION_URL,
                         json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"删除音色失败: {resp.text}")

    registry = _load_cosyvoice_registry()
    registry = [v for v in registry if v["voice_id"] != voice_id]
    _save_cosyvoice_registry(registry)
    return resp.json()


class CosyTTSRequest(BaseModel):
    text: str
    voice: str
    model: str = "cosyvoice-v3.5-plus"
    sample_rate: int = 8000
    mode: str = "line"
    speech_rate: float = 1.0
    pitch_rate: float = 1.0


@app.post("/api/cosyvoice/tts")
async def cosyvoice_tts(req: CosyTTSRequest):
    try:
        loop = asyncio.get_event_loop()
        wav_data = await loop.run_in_executor(
            None, _run_cosyvoice_tts, req.text, req.voice, req.model, req.sample_rate, req.mode, req.speech_rate, req.pitch_rate
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {e}")
    return Response(content=wav_data, media_type="audio/wav")
