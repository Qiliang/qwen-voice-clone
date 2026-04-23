"""
voice_filter.py

从 WAV 文件中提取人声片段并合并输出。

输入要求: sample_rate=8000, bit=16, mono
使用 silero_vad.onnx (onnxruntime) 进行人声检测
"""

import argparse
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnxruntime

# 模型默认路径
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"

# 8kHz 时每帧 256 个样本
FRAME_SIZE = 256
SAMPLE_RATE = 8000


class SileroOnnxModel:
    def __init__(self, path: str, force_onnx_cpu: bool = True):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()

    def reset_states(self, batch_size: int = 1):
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)

        batch_size = np.shape(x)[0]
        context_size = 32  # 8000Hz 时为 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": np.array(sr, dtype="int64"),
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs
        self._state = state
        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件，返回 int16 样本数组和采样率。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"仅支持 16-bit 音频，当前为 {sampwidth * 8}-bit")
    if n_channels != 1:
        raise ValueError(f"仅支持单声道音频，当前为 {n_channels} 声道")
    if sr != SAMPLE_RATE:
        raise ValueError(f"仅支持 {SAMPLE_RATE}Hz 采样率，当前为 {sr}Hz")

    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, sr


def write_wav(path: str, samples: np.ndarray, sr: int):
    """将 int16 样本写入 WAV 文件。"""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype(np.int16).tobytes())


def rms_db(samples: np.ndarray) -> float:
    """计算音频片段的 RMS 能量（dB）。"""
    if len(samples) == 0:
        return -120.0
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -120.0
    return 20.0 * np.log10(rms / 32768.0)


def extract_voice_segments(
    samples: np.ndarray,
    sr: int,
    model: SileroOnnxModel,
    vad_threshold: float = 0.5,
    min_volume_db: float = -40.0,
) -> list[tuple[int, int]]:
    """
    使用 Silero VAD 检测人声片段。

    返回 (start_sample, end_sample) 列表。
    """
    segments = []
    n_frames = len(samples) // FRAME_SIZE
    in_speech = False
    seg_start = 0

    for i in range(n_frames):
        frame = samples[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        audio_float32 = frame.astype(np.float32) / 32768.0

        confidence = model(audio_float32, sr)[0][0]
        volume = rms_db(frame)

        is_voice = confidence >= vad_threshold and volume >= min_volume_db

        if is_voice and not in_speech:
            in_speech = True
            seg_start = i * FRAME_SIZE
        elif not is_voice and in_speech:
            in_speech = False
            segments.append((seg_start, i * FRAME_SIZE))

    if in_speech:
        segments.append((seg_start, n_frames * FRAME_SIZE))

    return segments


def expand_segments(
    segments: list[tuple[int, int]],
    overlap_samples: int,
    total_samples: int,
) -> list[tuple[int, int]]:
    """
    将每个片段向前/向后各扩展 overlap_samples 个采样点，
    扩展后若相邻片段重叠则自动合并。

    Args:
        segments:        原始片段列表 [(start, end), ...]
        overlap_samples: 前后各扩展的采样点数
        total_samples:   音频总采样点数（用于边界裁剪）

    Returns:
        扩展并合并后的片段列表。
    """
    if not segments or overlap_samples <= 0:
        return segments

    expanded = [
        (max(0, s - overlap_samples), min(total_samples, e + overlap_samples))
        for s, e in segments
    ]

    # 合并相互重叠或相接的片段
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = expanded[0]
    for s, e in expanded[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    return merged


def merge_segments(
    samples: np.ndarray,
    segments: list[tuple[int, int]],
    gap_samples: int,
) -> np.ndarray:
    """
    将检测到的人声片段合并，片段之间插入静音间隔。

    gap_samples: 两段人声之间插入的静音样本数
    """
    if not segments:
        return np.array([], dtype=np.int16)

    silence = np.zeros(gap_samples, dtype=np.int16)
    parts = []
    for idx, (start, end) in enumerate(segments):
        parts.append(samples[start:end])
        if idx < len(segments) - 1:
            parts.append(silence)

    return np.concatenate(parts).astype(np.int16)


def voice_filter(
    input_path: str,
    output_path: str,
    model_path: str = str(DEFAULT_MODEL_PATH),
    gap_ms: int = 200,
    overlap_ms: int = 100,
    min_volume_db: float = -40.0,
    vad_threshold: float = 0.5,
):
    """
    主处理函数：读取音频 → VAD 检测 → Overlap 扩展 → 合并人声片段 → 写入输出。

    Args:
        input_path:    输入 WAV 文件路径
        output_path:   输出 WAV 文件路径
        model_path:    silero_vad.onnx 模型路径
        gap_ms:        相邻人声片段之间的静音间隔（毫秒）
        overlap_ms:    每个片段首尾各扩展的时长（毫秒），避免截断
        min_volume_db: 人声片段的最低音量阈值（dB），低于此值不计为人声
        vad_threshold: VAD 置信度阈值（0~1）
    """
    print(f"读取音频: {input_path}")
    samples, sr = read_wav(input_path)
    print(f"  样本数: {len(samples)}, 时长: {len(samples)/sr:.2f}s")

    print(f"加载模型: {model_path}")
    model = SileroOnnxModel(model_path)

    print(f"检测人声 (vad_threshold={vad_threshold}, min_volume_db={min_volume_db}dB)...")
    segments = extract_voice_segments(
        samples, sr, model,
        vad_threshold=vad_threshold,
        min_volume_db=min_volume_db,
    )
    print(f"  检测到 {len(segments)} 个人声片段")

    overlap_samples = int(sr * overlap_ms / 1000)
    segments = expand_segments(segments, overlap_samples, len(samples))
    print(f"  Overlap 扩展 ±{overlap_ms}ms 后: {len(segments)} 个片段")
    for i, (s, e) in enumerate(segments):
        print(f"    [{i+1}] {s/sr:.3f}s ~ {e/sr:.3f}s  ({(e-s)/sr*1000:.0f}ms)")

    gap_samples = int(sr * gap_ms / 1000)
    merged = merge_segments(samples, segments, gap_samples)
    print(f"合并后时长: {len(merged)/sr:.2f}s (间隔={gap_ms}ms)")

    write_wav(output_path, merged, sr)
    print(f"输出写入: {output_path}")


def wav_to_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: str = "192k") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，请先安装并确保在 PATH 中（例如: brew install ffmpeg）")
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(mp3_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 转 MP3 失败: {err}")


def main():
    dir_path = Path("/Users/xiaoql/Downloads/录音")
    output_dir = Path("/Users/xiaoql/Downloads/录音/output")
    file_pattern = "20200909*.wav"
    files = sorted(dir_path.glob(file_pattern))
    if not files:
        print(f"未找到匹配文件: {dir_path / file_pattern}")
        return

    model_path = str(DEFAULT_MODEL_PATH)
    gap_ms, overlap_ms = 100, 150
    min_volume_db, vad_threshold = -50, 0.6

    for inp in files:
        if not inp.is_file():
            continue
        out_mp3 = output_dir / f"{inp.stem}.mp3"
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            print(f"\n========== {inp.name} -> {out_mp3.name} ==========")
            voice_filter(
                input_path=str(inp),
                output_path=tmp_wav,
                model_path=model_path,
                gap_ms=gap_ms,
                overlap_ms=overlap_ms,
                min_volume_db=min_volume_db,
                vad_threshold=vad_threshold,
            )
            wav_to_mp3(tmp_wav, out_mp3)
            print(f"MP3 已写入: {out_mp3}")
        finally:
            Path(tmp_wav).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
