import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
import voice_extract
from encode import codec as cn_codec

app = FastAPI(title="Qwen Voice Clone")
logger = logging.getLogger("qwen-voice-clone")


def _encode_voice_prefix(
    name: str, max_len: int, *, alnum_only: bool = False
) -> str:
    """中文/数字音色名 → API prefix / preferred_name。"""
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="音色名称不能为空")
    try:
        return cn_codec.encode(name, max_len=max_len, alnum_only=alnum_only)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _cased_prefix_from_resource_link(
    resource_link: str, voice_id: str, fallback: str
) -> str:
    """resource_link 路径中仍保留原始大小写的 prefix（API 的 voice_id 会强制小写）。"""
    if not resource_link or not voice_id or not fallback:
        return fallback
    marker = f"/{voice_id}"
    idx = resource_link.find(marker)
    if idx <= 0:
        return fallback
    part = resource_link[:idx].rsplit("/", 1)[-1]
    if part and part.lower() == fallback.lower():
        return part
    return fallback


def _restore_underscore_tail(text: str, encoded_prefix: str) -> str:
    """alnum_only 编码会折叠 ``南枝_0`` → ``南枝0``；展示时若往返一致则还原 ``_``。"""
    if not text or "_" in text:
        return text
    match = re.fullmatch(r"^(.*[^\d])(\d+)$", text)
    if not match:
        return text
    candidate = f"{match.group(1)}_{match.group(2)}"
    try:
        limit = max(len(encoded_prefix), 16)
        if cn_codec.encode(candidate, max_len=limit, alnum_only=True) == encoded_prefix:
            return candidate
    except ValueError:
        pass
    return text


def _decode_voice_name(encoded: str) -> str | None:
    """将编码后的 prefix/preferred_name 还原为中文名。

    仅当含大写字母时解码：CnNameCodec 区分大小写，voice_id 会被 API 强制小写；
    全小写历史拼音 prefix 保持原样，避免短码碰撞出错误汉字。
    """
    if not encoded or not any(c.isupper() for c in encoded):
        return None
    try:
        text = cn_codec.decode_auto(encoded)
    except ValueError:
        return None
    return _restore_underscore_tail(text, encoded)

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
        print(f"[CosyVoiceCollector.on_error] {self.error}")
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

    first_delay_ms = None
    try:
        first_delay_ms = synthesizer.get_first_package_delay()
    except Exception:
        pass
    print(f"first_delay_ms: {first_delay_ms}")
    return _pcm_to_wav(src_pcm, sample_rate), first_delay_ms


def _qwen_audio_api_bases() -> tuple[str, str]:
    """返回 (http_base, websocket_base)。若配置了 Workspace ID 则走北京 MaaS 专属域名。"""
    workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
    if workspace_id:
        return (
            f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1",
            f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
        )
    return COSYVOICE_HTTP_URL, COSYVOICE_WS_URL


def _format_qwen_audio_tts_error(err: str) -> str:
    if "Model.AccessDenied" in err or "Model access denied" in err:
        return (
            "模型无访问权限 (Model.AccessDenied)。请在百炼控制台确认已开通 "
            "qwen-audio-3.0-tts-flash，并设置 DASHSCOPE_WORKSPACE_ID 后重试"
        )
    return err


def _run_qwen_audio_tts(
    text: str,
    voice: str,
    model: str,
    sample_rate: int = 8000,
    mode: str = "line",
    speech_rate: float = 1.0,
    pitch_rate: float = 1.0,
) -> tuple[bytes, float | None]:
    """Qwen-Audio 双向流式 TTS：streaming_call 推流，按采样率输出 PCM 再封装为 WAV。"""
    if mode == "line":
        chunks = [ln for ln in text.split("\n") if ln.strip()]
    else:
        chunks = list(text)
    if not chunks:
        raise ValueError("文本内容为空")

    if sample_rate == 16000:
        audio_format = CosyAudioFormat.PCM_16000HZ_MONO_16BIT
    elif sample_rate == 24000:
        audio_format = CosyAudioFormat.PCM_24000HZ_MONO_16BIT
    else:
        sample_rate = 8000
        audio_format = CosyAudioFormat.PCM_8000HZ_MONO_16BIT

    http_base, ws_base = _qwen_audio_api_bases()
    dashscope.base_http_api_url = http_base
    dashscope.base_websocket_api_url = ws_base

    collector = _CosyVoiceCollector()
    synthesizer = CosySpeechSynthesizer(
        model=model,
        voice=voice,
        format=audio_format,
        callback=collector,
        speech_rate=speech_rate,
        pitch_rate=pitch_rate,
    )
    try:
        # 双向流式：逐段 streaming_call，最后 streaming_complete
        for chunk in chunks:
            synthesizer.streaming_call(chunk)
        synthesizer.streaming_complete()
    except Exception as e:
        detail = collector.error or e
        print(
            f"[qwen-audio/tts] streaming 异常: model={model} voice={voice}\n"
            f"collector.error={collector.error!r}\n"
            f"exception={e!r}\n{traceback.format_exc()}"
        )
        if collector.error:
            raise RuntimeError(
                f"TTS 错误: {_format_qwen_audio_tts_error(str(collector.error))}"
            ) from e
        raise RuntimeError(f"TTS 错误: {_format_qwen_audio_tts_error(str(e))}") from e

    if not collector.wait(timeout=120):
        print(f"[qwen-audio/tts] 超时: model={model} voice={voice} collector.error={collector.error!r}")
        raise TimeoutError("Qwen-Audio TTS 超时")
    if collector.error:
        print(
            f"[qwen-audio/tts] collector 报错: model={model} voice={voice}\n"
            f"error={collector.error!r}"
        )
        raise RuntimeError(
            f"TTS 错误: {_format_qwen_audio_tts_error(str(collector.error))}"
        )

    pcm = collector.pcm_bytes()
    if not pcm:
        raise RuntimeError("TTS 未返回音频数据")

    first_delay_ms = None
    try:
        first_delay_ms = synthesizer.get_first_package_delay()
    except Exception:
        pass
    return _pcm_to_wav(pcm, sample_rate), first_delay_ms


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

    first_delay_ms = None
    try:
        first_delay_ms = client.get_first_audio_delay()
    except Exception:
        pass
    return _pcm_to_wav(pcm, sample_rate), first_delay_ms


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", dependencies=[Depends(_verify_basic)])
async def index():
    return FileResponse("static/index.html")


@app.get("/qwen", dependencies=[Depends(_verify_basic)])
async def qwen_index():
    return FileResponse("static/qwen-voice-clone.html")


def _extract_qwen_voice_prefix(voice_id: str) -> str:
    # voice ID: qwen-tts-vc-{preferred_name}-voice-{timestamp}-{hash}
    m = re.match(r"^qwen-tts-vc-(.+?)-voice-", voice_id or "")
    return m.group(1) if m else ""


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
    data = resp.json()
    for v in data.get("output", {}).get("voice_list", []) or []:
        voice_id = str(v.get("voice") or "")
        enc = v.get("preferred_name") or _extract_qwen_voice_prefix(voice_id)
        enc = str(enc or "")
        decoded = _decode_voice_name(enc)
        if decoded:
            v["preferred_name"] = decoded
            v["display_name"] = decoded
            v["prefix_encoded"] = enc
    return data


class CreateVoiceRequest(BaseModel):
    preferred_name: str
    audio_data: str  # base64 data URI，如 "data:audio/wav;base64,..."
    target_model: str = "qwen3-tts-vc-realtime-2026-01-15"


@app.post("/api/voices")
async def create_voice(req: CreateVoiceRequest):
    # preferred_name：仅允许数字/字母/下划线，最长 16 → 中文名先编码
    display_name = req.preferred_name.strip()
    encoded = _encode_voice_prefix(display_name, max_len=16)
    if not re.match(r"^[A-Za-z0-9_]{1,16}$", encoded):
        raise HTTPException(status_code=400, detail="编码后的 preferred_name 非法")
    payload = {
        "model": "qwen-voice-enrollment",
        "input": {
            "action": "create",
            "target_model": req.target_model,
            "preferred_name": encoded,
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
        wav_data, first_delay_ms = await loop.run_in_executor(
            None, _run_tts_ws, req.text, req.voice, req.model, req.sample_rate, req.mode, req.speech_rate, req.pitch_rate
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {e}")
    headers = {}
    if first_delay_ms is not None:
        headers["X-First-Package-Delay-Ms"] = str(first_delay_ms)
    return Response(content=wav_data, media_type="audio/wav", headers=headers)


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
    voice_list = [v for v in voice_list if str(v.get("voice_id", "")).startswith("cosyvoice")]
    voice_list.sort(key=lambda v: v.get("gmt_modified", ""), reverse=True)
    return {"voices": voice_list}


class CosyCreateVoiceRequest(BaseModel):
    voice_name: str
    audio_data: str  # base64 data URI: "data:audio/wav;base64,..."
    target_model: str = "cosyvoice-v3.5-plus"
    enable_preprocess: bool = True


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
            enable_preprocess=req.enable_preprocess,
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
        wav_data, first_delay_ms = await loop.run_in_executor(
            None, _run_cosyvoice_tts, req.text, req.voice, req.model, req.sample_rate, req.mode, req.speech_rate, req.pitch_rate
        )
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {e}")
    headers = {}
    if first_delay_ms is not None:
        headers["X-First-Package-Delay-Ms"] = str(first_delay_ms)
    return Response(content=wav_data, media_type="audio/wav", headers=headers)


# ===== Qwen-Audio Routes =====

_QWEN_AUDIO_MODELS = (
    "qwen-audio-3.0-tts-flash",
    "qwen-audio-3.0-tts-plus",
)


def _extract_qwen_audio_prefix(voice_id: str) -> tuple[str, str]:
    """从 voice_id 解析 (target_model, prefix)。解析失败返回 ('', '')。"""
    for model in _QWEN_AUDIO_MODELS:
        model_prefix = model + "-"
        if not voice_id.startswith(model_prefix):
            continue
        rest = voice_id[len(model_prefix):]
        name, _, _uid = rest.rpartition("-")
        return model, name
    return "", ""


def _query_qwen_audio_resource_link(voice_id: str) -> str:
    """查询单个音色以拿到含原始大小写 prefix 的 resource_link。"""
    try:
        payload = {
            "model": "voice-enrollment",
            "input": {"action": "query_voice", "voice_id": voice_id},
        }
        resp = requests.post(
            COSYVOICE_CUSTOMIZATION_URL, json=payload, headers=get_headers(), timeout=15
        )
        if resp.status_code != 200:
            return ""
        return str(resp.json().get("output", {}).get("resource_link") or "")
    except Exception:
        return ""


def _enrich_qwen_audio_voice(v: dict) -> dict:
    """从 voice_id 解析 display_name / target_model（格式: {model}-{prefix}-{uuid}）。

    prefix 经 CnNameCodec 编码；尽量用 resource_link 中的原始大小写再解码。
    """
    voice_id = str(v.get("voice_id", ""))
    item = dict(v)
    model, prefix = _extract_qwen_audio_prefix(voice_id)
    if model:
        item["target_model"] = model
    else:
        item.setdefault("target_model", "qwen-audio-3.0-tts-flash")

    if prefix:
        cased = _cased_prefix_from_resource_link(
            str(item.get("resource_link") or ""),
            voice_id,
            prefix,
        )
        item["prefix_encoded"] = cased
        item["display_name"] = _decode_voice_name(cased) or cased
    return item


@app.get("/qwen-audio", dependencies=[Depends(_verify_basic)])
async def qwen_audio_index():
    return FileResponse("static/qwen-audio-clone.html")


@app.get("/api/qwen-audio/voices")
async def qwen_audio_list_voices():
    payload = {
        "model": "voice-enrollment",
        "input": {"action": "list_voice", "page_size": 1000, "page_index": 0},
    }
    resp = requests.post(COSYVOICE_CUSTOMIZATION_URL, json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"API 错误: {resp.text}")
    data = resp.json()
    raw_list = [
        v for v in data.get("output", {}).get("voice_list", []) or []
        if str(v.get("voice_id", "")).startswith("qwen-audio")
    ]

    # list 不含 resource_link；补查以恢复 prefix 大小写后再解码
    def _enrich_all() -> list[dict]:
        def _with_link(v: dict) -> dict:
            voice_id = str(v.get("voice_id", ""))
            link = _query_qwen_audio_resource_link(voice_id)
            if link:
                v = {**v, "resource_link": link}
            return _enrich_qwen_audio_voice(v)

        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(_with_link, raw_list))

    loop = asyncio.get_event_loop()
    voice_list = await loop.run_in_executor(None, _enrich_all)
    voice_list.sort(key=lambda v: v.get("gmt_modified", ""), reverse=True)
    return {"voices": voice_list}


class QwenAudioCreateVoiceRequest(BaseModel):
    voice_name: str
    audio_data: str  # base64 data URI: "data:audio/wav;base64,..."
    target_model: str = "qwen-audio-3.0-tts-flash"
    language_hints: list[str] | None = None


@app.post("/api/qwen-audio/voices")
async def qwen_audio_create_voice(req: QwenAudioCreateVoiceRequest):
    # prefix：仅允许数字和英文字母，最长 10 → 中文名先编码（_ 后缀折叠，不出现在 prefix）
    display_name = req.voice_name.strip()
    encoded = _encode_voice_prefix(display_name, max_len=10, alnum_only=True)

    try:
        header, b64 = req.audio_data.split(",", 1)
        audio_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="音频数据格式错误")

    tmp_filename = f"{encoded}_{int(time.time())}.wav"
    tmp_path = UPLOAD_DIR / tmp_filename
    tmp_path.write_bytes(audio_bytes)

    try:
        public_url = rustfs_upload(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传到公网失败: {e}")

    dashscope.base_http_api_url = COSYVOICE_HTTP_URL
    dashscope.base_websocket_api_url = COSYVOICE_WS_URL

    service = VoiceEnrollmentService()
    create_kwargs = {
        "target_model": req.target_model,
        "prefix": encoded,
        "url": public_url,
    }
    if req.language_hints:
        create_kwargs["language_hints"] = req.language_hints
    try:
        voice_id = service.create_voice(**create_kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建音色失败: {e}")

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

    return {"voice_id": voice_id, "status": status, "display_name": display_name}


@app.get("/api/qwen-audio/voices/{voice_id:path}/status")
async def qwen_audio_voice_status(voice_id: str):
    dashscope.base_http_api_url = COSYVOICE_HTTP_URL
    service = VoiceEnrollmentService()
    try:
        info = service.query_voice(voice_id=voice_id)
        status = info.get("status", "DEPLOYING")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"voice_id": voice_id, "status": status}


@app.delete("/api/qwen-audio/voices/{voice_id:path}")
async def qwen_audio_delete_voice(voice_id: str):
    payload = {
        "model": "voice-enrollment",
        "input": {"action": "delete_voice", "voice_id": voice_id},
    }
    resp = requests.post(COSYVOICE_CUSTOMIZATION_URL,
                         json=payload, headers=get_headers())
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"删除音色失败: {resp.text}")
    return resp.json()


class QwenAudioTTSRequest(BaseModel):
    text: str
    voice: str
    model: str = "qwen-audio-3.0-tts-flash"
    sample_rate: int = 8000
    mode: str = "line"
    speech_rate: float = 1.0
    pitch_rate: float = 1.0


@app.post("/api/qwen-audio/tts")
async def qwen_audio_tts(req: QwenAudioTTSRequest):
    try:
        loop = asyncio.get_event_loop()
        wav_data, first_delay_ms = await loop.run_in_executor(
            None,
            _run_qwen_audio_tts,
            req.text,
            req.voice,
            req.model,
            req.sample_rate,
            req.mode,
            req.speech_rate,
            req.pitch_rate,
        )
    except TimeoutError as e:
        logger.exception(
            "Qwen-Audio TTS 超时: model=%s voice=%s sample_rate=%s mode=%s",
            req.model, req.voice, req.sample_rate, req.mode,
        )
        print(f"[qwen-audio/tts] TimeoutError: {e!r}\n{traceback.format_exc()}")
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception(
            "Qwen-Audio TTS 失败: model=%s voice=%s sample_rate=%s mode=%s err=%s",
            req.model, req.voice, req.sample_rate, req.mode, e,
        )
        print(
            f"[qwen-audio/tts] ERROR model={req.model} voice={req.voice} "
            f"sample_rate={req.sample_rate} mode={req.mode}\n"
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"TTS 合成失败: {e}")
    headers = {}
    if first_delay_ms is not None:
        headers["X-First-Package-Delay-Ms"] = str(first_delay_ms)
    return Response(content=wav_data, media_type="audio/wav", headers=headers)


# ===== MiniMax Routes =====

MINIMAX_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MINIMAX_REGISTRY_FILE = pathlib.Path("uploads/minimax_demo_registry.json")


def _load_minimax_registry() -> list:
    if MINIMAX_REGISTRY_FILE.exists():
        try:
            return json.loads(MINIMAX_REGISTRY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_minimax_registry(registry: list) -> None:
    MINIMAX_REGISTRY_FILE.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def get_minimax_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }


@app.get("/minimax", dependencies=[Depends(_verify_basic)])
async def minimax_index():
    return FileResponse("static/minimax-voice-clone.html")


class MiniMaxCloneRequest(BaseModel):
    voice_id: str
    text: str
    audio_data: str  # base64 data URI: "data:audio/wav;base64,..."
    model: str = "MiniMax/speech-2.8-turbo"
    language_boost: str | None = None
    need_noise_reduction: bool = False
    need_volume_normalization: bool = False


@app.post("/api/minimax/clone")
async def minimax_clone(req: MiniMaxCloneRequest):
    # voice_id: [8,256], first letter, alnum/-/_, last not -/_
    if not re.match(r"^[A-Za-z][A-Za-z0-9_-]{6,254}[A-Za-z0-9]$", req.voice_id):
        raise HTTPException(
            status_code=400,
            detail="voice_id 需 8~256 位：首字母、末位非 -/_，仅字母数字-_",
        )
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="试听文本不能为空")
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="试听文本不能超过 1000 字符")

    allowed_models = {
        "MiniMax/speech-2.8-hd",
        "MiniMax/speech-02-hd",
        "MiniMax/speech-2.8-turbo",
        "MiniMax/speech-02-turbo",
    }
    if req.model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {req.model}")

    try:
        header, b64 = req.audio_data.split(",", 1)
        audio_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="音频数据格式错误")

    if len(audio_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="音频文件不能超过 20MB")

    tmp_filename = f"minimax_{req.voice_id}_{int(time.time())}.wav"
    tmp_path = UPLOAD_DIR / tmp_filename
    tmp_path.write_bytes(audio_bytes)

    try:
        public_url = rustfs_upload(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"上传到公网失败: {e}")

    input_payload: dict = {
        "action": "voice_clone",
        "voice_id": req.voice_id,
        "audio_url": public_url,
        "text": text,
        "need_noise_reduction": req.need_noise_reduction,
        "need_volume_normalization": req.need_volume_normalization,
    }
    if req.language_boost:
        input_payload["language_boost"] = req.language_boost

    payload = {"model": req.model, "input": input_payload}

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                MINIMAX_URL, json=payload, headers=get_minimax_headers(), timeout=120
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用 MiniMax API 失败: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"复刻失败: {resp.text}")

    data = resp.json()
    output = data.get("output") or {}
    base_resp = output.get("base_resp") or {}
    status_code = base_resp.get("status_code", -1)
    if status_code != 0:
        raise HTTPException(
            status_code=500,
            detail=f"复刻失败 [{status_code}]: {base_resp.get('status_msg', 'unknown')}",
        )

    demo_audio = output.get("demo_audio")
    if not demo_audio:
        raise HTTPException(status_code=500, detail="未返回试听音频")

    entry = {
        "id": f"{req.voice_id}_{int(time.time())}",
        "voice_id": req.voice_id,
        "model": req.model,
        "text": text,
        "demo_audio": demo_audio,
        "audio_url": public_url,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_id": data.get("request_id"),
        "characters": (data.get("usage") or {}).get("characters"),
    }
    registry = _load_minimax_registry()
    registry.insert(0, entry)
    _save_minimax_registry(registry)
    return {"demo_audio": demo_audio, "voice_id": req.voice_id, "entry": entry}


@app.get("/api/minimax/demos")
async def minimax_list_demos():
    return {"demos": _load_minimax_registry()}


def _audio_bytes_to_8k_wav(audio_bytes: bytes) -> bytes:
    """解码任意常见音频并重采样为 8kHz mono WAV。"""
    import librosa

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        if sr != 8000:
            y = soxr.resample(y, sr, 8000)
        y_i16 = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
        return _pcm_to_wav(y_i16.tobytes(), 8000)
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/minimax/play-8k")
async def minimax_play_8k(url: str):
    """代理拉取试听音频并转成 8k WAV，供前端播放。"""
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="无效的音频 URL")
    try:
        loop = asyncio.get_event_loop()

        def _fetch_and_convert():
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(f"下载失败: HTTP {resp.status_code}")
            return _audio_bytes_to_8k_wav(resp.content)

        wav_data = await loop.run_in_executor(None, _fetch_and_convert)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转 8k 失败: {e}")
    return Response(content=wav_data, media_type="audio/wav")


@app.delete("/api/minimax/demos/{demo_id}")
async def minimax_delete_demo(demo_id: str):
    registry = _load_minimax_registry()
    new_registry = [d for d in registry if d.get("id") != demo_id]
    if len(new_registry) == len(registry):
        raise HTTPException(status_code=404, detail="试听记录不存在")
    _save_minimax_registry(new_registry)
    return {"ok": True}


# ===== 视频音频提取 Routes =====

@app.get("/extract", dependencies=[Depends(_verify_basic)])
async def extract_index():
    return FileResponse("static/voice-extract.html")


@app.get("/api/extract/cookies")
async def extract_get_cookies():
    return {"cookies": voice_extract.get_default_cookies()}


class ExtractCookiesRequest(BaseModel):
    cookies: str


@app.post("/api/extract/cookies")
async def extract_save_cookies(req: ExtractCookiesRequest):
    voice_extract.save_default_cookies(req.cookies)
    return {"ok": True}


class ExtractRequest(BaseModel):
    url: str
    time_range: str
    audio_format: str = "wav"  # mp3 | wav | m4a
    cookies: str | None = None
    save_cookies: bool = False
    do_vocal: bool = False
    do_diarize: bool = False
    diarize_num_speakers: int | None = None
    do_vad: bool = False
    vad_overlap_ms: int = 200


@app.post("/api/extract", dependencies=[Depends(_verify_basic)])
async def extract_run(req: ExtractRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            voice_extract.run_pipeline,
            req.url,
            req.time_range,
            req.audio_format,
            req.cookies,
            req.save_cookies,
            req.do_vocal,
            req.do_diarize,
            req.diarize_num_speakers,
            req.do_vad,
            req.vad_overlap_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频提取失败: {e}")
    return result


@app.get("/api/extract/files/{task_id}/{filename}")
async def extract_get_file(task_id: str, filename: str):
    try:
        path = voice_extract.get_extract_file(task_id, filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(str(path), filename=filename)
