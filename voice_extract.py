"""视频音频提取流水线。

流程（每步可选，但步骤 1 必选）：
1. yt-dlp 按时间段下载音频
2. Demucs htdemucs 人声分离
3. SpeechBrain ECAPA-TDNN + 聚类做说话人分离
4. silero-vad 静音裁剪（保留 overlap 边界）

所有步骤同步串行，FastAPI 端用 run_in_executor 包装。
"""

from __future__ import annotations

import os
import pathlib
import re
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf

EXTRACTS_DIR = pathlib.Path("extracts")
EXTRACTS_DIR.mkdir(exist_ok=True)
COOKIES_DEFAULT_FILE = pathlib.Path("cookies.txt")
SPEECHBRAIN_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"

# 模型懒加载
_demucs_separator = None
_vad_model = None
_speaker_encoder = None

# 时间段格式 [*]HH:MM:SS-HH:MM:SS，星号是 yt-dlp 的绝对时间戳约定，可选
_TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})-(\d{1,2}):(\d{2}):(\d{2})$")


# ---------- 工具 ----------

def _parse_time_range(s: str) -> tuple[float, float]:
    raw = (s or "").strip().lstrip("*")
    m = _TIME_RANGE_RE.match(raw)
    if not m:
        raise ValueError(f"时间段格式错误，期望 HH:MM:SS-HH:MM:SS，实际：{s!r}")
    h1, m1, s1, h2, m2, s2 = map(int, m.groups())
    start = h1 * 3600 + m1 * 60 + s1
    end = h2 * 3600 + m2 * 60 + s2
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    return float(start), float(end)


def get_default_cookies() -> str:
    if COOKIES_DEFAULT_FILE.exists():
        return COOKIES_DEFAULT_FILE.read_text(encoding="utf-8")
    return ""


def save_default_cookies(cookies_text: str) -> None:
    COOKIES_DEFAULT_FILE.write_text(cookies_text or "", encoding="utf-8")


# ---------- Step 1: yt-dlp ----------

def _yt_dlp_download(
    url: str,
    time_range: str,
    audio_format: str,
    cookies_path: Optional[pathlib.Path],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    start, end = _parse_time_range(time_range)
    out_template = str(out_dir / "original.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "overwrites": True,
    }
    if cookies_path is not None and cookies_path.exists():
        opts["cookiefile"] = str(cookies_path)

    with YoutubeDL(opts) as ydl:
        ydl.download([url])

    target = out_dir / f"original.{audio_format}"
    if target.exists():
        return target
    # postprocessor 偶尔会留下原始容器后缀，做一次兜底
    candidates = [p for p in out_dir.glob("original.*") if p.suffix.lower() != ".txt"]
    if not candidates:
        raise RuntimeError("yt-dlp 下载失败：未生成音频文件")
    return candidates[0]


# ---------- Step 2: Demucs 人声分离 ----------

def _load_demucs():
    """加载 Demucs htdemucs 模型（lazy）。

    用底层 demucs.pretrained / demucs.apply API，避开 demucs 4.0.1 PyPI 还没发布
    demucs.api 的问题（demucs.api 仅在 master 分支存在）。
    """
    global _demucs_separator
    if _demucs_separator is None:
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        model.cpu().eval()
        _demucs_separator = model
    return _demucs_separator


def _demucs_separate_vocals(input_path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    import librosa
    import torch
    from demucs.apply import apply_model

    model = _load_demucs()
    target_sr = int(model.samplerate)
    target_ch = int(model.audio_channels)

    # 用 librosa 读取，自动重采样到 Demucs 期望的 44.1k；保留多通道
    wav_np, _ = librosa.load(
        str(input_path), sr=target_sr, mono=False
    )  # mono=(samples,), stereo=(channels, samples)
    if wav_np.ndim == 1:
        wav_np = np.stack([wav_np, wav_np], axis=0)  # mono -> stereo
    if wav_np.shape[0] > target_ch:
        wav_np = wav_np[:target_ch]
    elif wav_np.shape[0] == 1 and target_ch == 2:
        wav_np = np.concatenate([wav_np, wav_np], axis=0)
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float()

    # 标准化（Demucs 官方 separate.py 的做法），分离后再还原
    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std()
    if std < 1e-8:
        std = torch.tensor(1.0)
    wav_norm = (wav - mean) / std

    with torch.no_grad():
        sources = apply_model(
            model,
            wav_norm.unsqueeze(0),
            device="cpu",
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    if "vocals" not in model.sources:
        raise RuntimeError(f"Demucs 模型未提供 vocals 分轨：{model.sources}")
    idx = model.sources.index("vocals")
    vocals = sources[idx].cpu().numpy()  # (channels, samples)
    vocals_t = vocals.T if vocals.ndim == 2 else vocals
    out = out_dir / "vocals.wav"
    sf.write(str(out), vocals_t, target_sr, subtype="PCM_16")
    return out


# ---------- Step 3: silero-vad ----------

def _load_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    return _vad_model


def _load_audio_16k_mono(path: pathlib.Path) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(str(path), sr=16000, mono=True)
    return wav.astype(np.float32, copy=False)


def _vad_speech_segments(wav_np: np.ndarray) -> list[dict]:
    """Return silero-vad timestamps in samples at 16k."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_vad()
    wav_t = torch.from_numpy(wav_np)
    return get_speech_timestamps(wav_t, model, sampling_rate=16000)


def _vad_trim_file(
    input_path: pathlib.Path,
    out_path: pathlib.Path,
    overlap_ms: int,
) -> pathlib.Path:
    wav = _load_audio_16k_mono(input_path)
    ts = _vad_speech_segments(wav)
    if not ts:
        # 检测不到语音：写一份空音频，避免下游异常
        sf.write(str(out_path), np.zeros(0, dtype=np.float32), 16000, subtype="PCM_16")
        return out_path
    pad = max(0, int(overlap_ms / 1000.0 * 16000))
    chunks = []
    for seg in ts:
        s = max(0, seg["start"] - pad)
        e = min(len(wav), seg["end"] + pad)
        if e > s:
            chunks.append(wav[s:e])
    out_wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sf.write(str(out_path), out_wav, 16000, subtype="PCM_16")
    return out_path


# ---------- Step 4 (实际是 Step 3): 说话人分离 ----------

def _load_speaker_encoder():
    global _speaker_encoder
    if _speaker_encoder is None:
        # 关掉 HF symlink 警告（Windows 常见），并强制 COPY 策略避免 Win 普通用户没权限建符号链接
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        _speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SPEECHBRAIN_SAVEDIR,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
    return _speaker_encoder


def _diarize_file(
    input_path: pathlib.Path,
    out_dir: pathlib.Path,
    num_speakers: Optional[int] = None,
) -> list[pathlib.Path]:
    import torch
    from sklearn.cluster import AgglomerativeClustering

    encoder = _load_speaker_encoder()
    wav = _load_audio_16k_mono(input_path)
    ts = _vad_speech_segments(wav)
    if not ts:
        return []

    # 把段落分成可嵌入（>=0.5s）和过短（<0.5s）两组
    # 过短段落不算 ECAPA 嵌入（信息量不足），但稍后按时间最近邻挂到某个说话人，
    # 避免一个人的声音被切散到多个文件。
    min_samples = int(0.5 * 16000)
    wav_t = torch.from_numpy(wav)

    embeddable_ts: list[dict] = []
    short_ts: list[dict] = []
    for seg in ts:
        if seg["end"] - seg["start"] >= min_samples:
            embeddable_ts.append(seg)
        else:
            short_ts.append(seg)

    if not embeddable_ts:
        # 没有足够长的段：把所有短段都视作单一说话人
        if not short_ts:
            return []
        out_path = out_dir / "speaker_1.wav"
        chunks = [wav[s["start"] : s["end"]] for s in short_ts]
        out_wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        sf.write(str(out_path), out_wav, 16000, subtype="PCM_16")
        return [out_path]

    embs: list[np.ndarray] = []
    for seg in embeddable_ts:
        s, e = seg["start"], seg["end"]
        chunk = wav_t[s:e].unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(chunk)
        emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        embs.append(emb)

    embs_arr = np.stack(embs)

    if num_speakers and num_speakers > 0:
        if num_speakers == 1 or len(embs_arr) == 1:
            labels = np.zeros(len(embs_arr), dtype=int)
        else:
            n_clusters = min(num_speakers, len(embs_arr))
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
            labels = clusterer.fit_predict(embs_arr)
    else:
        if len(embs_arr) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            # 阈值越大、越倾向于把相近段合并到同一说话人。0.7 比 0.5 更保守，
            # 优先保证"同一人声进同一个文件"，宁可漏检不愿误分。
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.7,
                metric="cosine",
                linkage="average",
            )
            labels = clusterer.fit_predict(embs_arr)

    # 给每个原始 ts 段分配 label：长段直接用聚类结果；短段挂到时间最近的长段
    seg_to_label: dict[int, int] = {}
    for seg, lbl in zip(embeddable_ts, labels):
        seg_to_label[id(seg)] = int(lbl)

    def _nearest_label(seg: dict) -> int:
        mid = (seg["start"] + seg["end"]) / 2.0
        nearest = min(
            embeddable_ts,
            key=lambda e: abs((e["start"] + e["end"]) / 2.0 - mid),
        )
        return seg_to_label[id(nearest)]

    # 把所有段（含短段）按时间顺序聚到对应说话人，保持时序一致
    groups: dict[int, list[dict]] = {}
    for seg in ts:
        lbl = seg_to_label.get(id(seg))
        if lbl is None:
            lbl = _nearest_label(seg)
        groups.setdefault(lbl, []).append(seg)

    # 按首次出现时间重命名 speaker_1, speaker_2 ...
    ordered = sorted(groups.keys(), key=lambda k: groups[k][0]["start"])
    out_paths: list[pathlib.Path] = []
    for new_idx, lbl in enumerate(ordered, start=1):
        segs = sorted(groups[lbl], key=lambda s: s["start"])
        chunks = [wav[seg["start"] : seg["end"]] for seg in segs]
        out_wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out_path = out_dir / f"speaker_{new_idx}.wav"
        sf.write(str(out_path), out_wav, 16000, subtype="PCM_16")
        out_paths.append(out_path)
    return out_paths


# ---------- 顶层 pipeline ----------

def run_pipeline(
    url: str,
    time_range: str,
    audio_format: str,
    cookies_text: Optional[str] = None,
    save_cookies: bool = False,
    do_vocal: bool = False,
    do_diarize: bool = False,
    diarize_num_speakers: Optional[int] = None,
    do_vad: bool = False,
    vad_overlap_ms: int = 200,
) -> dict:
    if audio_format not in ("mp3", "wav", "m4a"):
        raise ValueError("audio_format 必须是 mp3 / wav / m4a")
    if vad_overlap_ms < 0 or vad_overlap_ms > 5000:
        raise ValueError("vad_overlap_ms 取值范围 0-5000")

    task_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    task_dir = EXTRACTS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    cookies_path: Optional[pathlib.Path] = None
    if cookies_text and cookies_text.strip():
        cookies_path = task_dir / "cookies.txt"
        cookies_path.write_text(cookies_text, encoding="utf-8")
        if save_cookies:
            save_default_cookies(cookies_text)

    files_meta: list[dict] = []

    def _meta(p: pathlib.Path, kind: str) -> dict:
        return {
            "name": p.name,
            "url": f"/api/extract/files/{task_id}/{p.name}",
            "kind": kind,
            "size": p.stat().st_size if p.exists() else 0,
        }

    # Step 1
    original_path = _yt_dlp_download(
        url, time_range, audio_format, cookies_path, task_dir
    )
    files_meta.append(_meta(original_path, "original"))

    # Step 2
    vocals_path: Optional[pathlib.Path] = None
    if do_vocal:
        vocals_path = _demucs_separate_vocals(original_path, task_dir)
        files_meta.append(_meta(vocals_path, "vocals"))

    # 下游说话人分离 / VAD 的输入：优先 vocals，否则 original
    diarize_input = vocals_path if vocals_path else original_path

    # Step 3
    speaker_paths: list[pathlib.Path] = []
    if do_diarize:
        speaker_paths = _diarize_file(
            diarize_input, task_dir, num_speakers=diarize_num_speakers
        )
        for p in speaker_paths:
            files_meta.append(_meta(p, "speaker"))

    # Step 4
    if do_vad:
        if speaker_paths:
            trim_inputs = speaker_paths
        else:
            trim_inputs = [diarize_input]
        for src in trim_inputs:
            stem = src.stem
            out = task_dir / f"trimmed_{stem}.wav"
            _vad_trim_file(src, out, overlap_ms=vad_overlap_ms)
            files_meta.append(_meta(out, "trimmed"))

    return {"task_id": task_id, "files": files_meta}


def get_extract_file(task_id: str, filename: str) -> pathlib.Path:
    """Return safe absolute path under extracts/<task_id>/<filename> or raise."""
    safe_task = re.sub(r"[^A-Za-z0-9_\-]", "", task_id)
    safe_name = pathlib.Path(filename).name  # 防 path traversal
    if not safe_task or not safe_name:
        raise FileNotFoundError(f"非法路径: {task_id}/{filename}")
    target = (EXTRACTS_DIR / safe_task / safe_name).resolve()
    base = EXTRACTS_DIR.resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise FileNotFoundError(f"路径越界: {target}") from e
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"文件不存在: {target}")
    return target
