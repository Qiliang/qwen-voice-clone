#!/usr/bin/env python3
"""从现有克隆音色的原始音频，创建新的 qwen-audio 声音复刻。

流程：
1. query_voice 获取原音色的 resource_link（原始音频 URL）与 target_model
2. 将中文 voice_name 编码为 API prefix
3. create_voice 用同一音频 URL 创建新音色

输入：原 voice_id，新 voice_name（中文名，最多 5 个字）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encode import codec as cn_codec

CUSTOMIZATION_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
)
DEFAULT_TARGET_MODEL = "qwen-audio-3.0-tts-flash"
VOICE_NAME_MAX_CHARS = 5  # 全小写 base36 + 1295 字表，prefix≤10 保证 5 字
PREFIX_MAX_LEN = 10


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise SystemExit("未设置 DASHSCOPE_API_KEY（可写在项目根目录 .env）")
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _post(api_key: str, payload: dict) -> dict:
    resp = requests.post(
        CUSTOMIZATION_URL,
        json=payload,
        headers=_headers(api_key),
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    if data.get("code") and str(data.get("code")) not in ("", "Success"):
        raise RuntimeError(f"API 错误: {data}")
    return data


def query_voice(api_key: str, voice_id: str) -> dict:
    """查询音色详情，返回含 resource_link / target_model / status 的 output。"""
    data = _post(
        api_key,
        {
            "model": "voice-enrollment",
            "input": {"action": "query_voice", "voice_id": voice_id},
        },
    )
    output = data.get("output") or {}
    if not output.get("resource_link"):
        raise RuntimeError(f"未拿到原始音频 URL（resource_link）: {data}")
    return output


def encode_prefix(voice_name: str) -> str:
    name = voice_name.strip()
    if not name:
        raise ValueError("voice_name 不能为空")
    if len(name) > VOICE_NAME_MAX_CHARS:
        raise ValueError(
            f"voice_name 最多 {VOICE_NAME_MAX_CHARS} 个字，当前 {len(name)} 个"
        )
    return cn_codec.encode(name, max_len=PREFIX_MAX_LEN, alnum_only=True)


def create_voice(
    api_key: str,
    *,
    url: str,
    prefix: str,
    target_model: str,
    language_hints: list[str] | None = None,
) -> str:
    """创建音色，返回新 voice_id。"""
    input_body: dict = {
        "action": "create_voice",
        "target_model": target_model,
        "prefix": prefix,
        "url": url,
        "language_hints": language_hints or ["zh"],
    }
    data = _post(
        api_key,
        {"model": "voice-enrollment", "input": input_body},
    )
    voice_id = (data.get("output") or {}).get("voice_id")
    if not voice_id:
        raise RuntimeError(f"创建成功但未返回 voice_id: {data}")
    return str(voice_id)


def wait_until_ready(
    api_key: str, voice_id: str, *, attempts: int = 12, interval: float = 5.0
) -> str:
    status = "DEPLOYING"
    for i in range(attempts):
        time.sleep(interval)
        try:
            info = query_voice(api_key, voice_id)
            status = str(info.get("status") or "DEPLOYING")
            print(f"  [{i + 1}/{attempts}] status={status}")
            if status in ("OK", "UNDEPLOYED"):
                return status
        except Exception as e:
            print(f"  [{i + 1}/{attempts}] 查询失败: {e}")
    return status


def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="基于已有 qwen-audio 音色的原始音频，创建新的声音复刻"
    # )
    # parser.add_argument("voice_id", help="原音色 voice_id")
    # parser.add_argument(
    #     "voice_name",
    #     help=f"新音色中文名（最多 {VOICE_NAME_MAX_CHARS} 个字）",
    # )
    # parser.add_argument(
    #     "--target-model",
    #     default=None,
    #     help=f"目标合成模型（默认沿用原音色，否则 {DEFAULT_TARGET_MODEL}）",
    # )
    # parser.add_argument(
    #     "--no-wait",
    #     action="store_true",
    #     help="创建后不等待审核完成",
    # )
    # args = parser.parse_args()

    _load_dotenv()
    api_key = _api_key()
    voice_id = "qwen-audio-3.0-tts-flash-xiaoying-0b1a4a61a1284c58a4619e8c5189700b"
    display_name = "小樱温柔"
    try:
        prefix = encode_prefix(display_name)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    print(f"原 voice_id : {voice_id}")
    print(f"新 voice_name: {display_name} → prefix={prefix}")

    print("\n--- 查询原音色 ---")
    info = query_voice(api_key, voice_id)
    resource_link = str(info["resource_link"])
    target_model = (
        "qwen-audio-3.0-tts-flash"
        or str(info.get("target_model") or "").strip()
        or DEFAULT_TARGET_MODEL
    )
    print(f"status       : {info.get('status')}")
    print(f"target_model : {target_model}")
    print(f"resource_link: {resource_link}")

    print("\n--- 创建新音色 ---")
    new_voice_id = create_voice(
        api_key,
        url=resource_link,
        prefix=prefix,
        target_model=target_model,
    )
    print(f"新 voice_id  : {new_voice_id}")


    print("\n--- 等待审核 ---")
    status = wait_until_ready(api_key, new_voice_id)
    print(f"\n完成: voice_id={new_voice_id} status={status} display_name={display_name}")


if __name__ == "__main__":
    main()
