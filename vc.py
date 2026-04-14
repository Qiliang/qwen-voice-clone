# coding=utf-8
# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio

import pyaudio
import os
import requests
import base64
import pathlib
import threading
import time
import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat
# ======= 常量配置 =======
DEFAULT_TARGET_MODEL = "qwen3-tts-vc-realtime-2026-01-15"  # 声音复刻、语音合成要使用相同的模型
# DEFAULT_TARGET_MODEL = "qwen3-tts-vc-realtime-2025-11-27"  # 声音复刻、语音合成要使用相同的模型
DEFAULT_PREFERRED_NAME = "malisu"
DEFAULT_AUDIO_MIME_TYPE = "audio/mpeg"
VOICE_FILE_PATH = "original/玛丽苏文学.mp3"  # 用于声音复刻的本地音频文件的相对路径


readme = """
音频3月5日燕燕.MP3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414103341204-340b'}, 'usage': {'count': 1}, 'request_id': 'ca27599a-3d53-90d5-b896-b5e9bdd96383'}
2月27日彬彬.mp3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414111114628-4ede'}, 'usage': {'count': 1}, 'request_id': '2754f684-c4b2-9922-a340-915ff92a8441'}
白皮书.mp3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414111228720-abad'}, 'usage': {'count': 1}, 'request_id': '795b0d31-5b84-9199-8115-49b7c21cce6b'}
讲小说内容.mp3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414112030260-cb3b'}, 'usage': {'count': 1}, 'request_id': '2a466070-7ddf-932f-aa7e-1d9f3c0663ba'}
量变到质变.mp3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414112116098-bc55'}, 'usage': {'count': 1}, 'request_id': '69eef172-41f6-941f-bfde-241043daf368'}
玛丽苏文学.mp3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414112425821-6654'}, 'usage': {'count': 1}, 'request_id': 'ba63842a-4aa6-9a43-a364-4258233ed62c'}
燕燕2月28日.MP3
{'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414113024788-4b84'}, 'usage': {'count': 1}, 'request_id': '903f867e-3e5b-96a7-9bc2-7067ea4c9c7e'}
云福.mp3
pass
"""


TEXT_TO_SYNTHESIZE = [
    '对吧~我就特别喜欢这种超市，',
    '尤其是过年的时候',
    '去逛超市',
    '就会觉得',
    '超级超级开心！',
    '想买好多好多的东西呢！'
]

# TEXT_TO_SYNTHESIZE = [
#     '具体而非模糊：使用能够描绘具体声音特质的词语，',
#     '如“低沉”、“清脆”、“语速偏快”。',
#     '避免使用“好听”、“普通”等主观且缺乏信息量的词汇。',
# ]
# TEXT_TO_SYNTHESIZE = [
#     '有声书和广播剧配音。',
#     '广告和宣传片配音。',
#     '游戏角色和动画配音。',
#     '情感化的智能语音助手。',
#     '纪录片和新闻播报。',
# ]


def create_voice(
        file_path: str,
        target_model: str = DEFAULT_TARGET_MODEL,
        preferred_name: str = DEFAULT_PREFERRED_NAME,
        audio_mime_type: str = DEFAULT_AUDIO_MIME_TYPE) -> str:
    """
    创建音色，并返回 voice 参数
    """
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx"
    api_key = os.getenv("DASHSCOPE_API_KEY")

    file_path_obj = pathlib.Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"音频文件不存在: {file_path}")

    base64_str = base64.b64encode(file_path_obj.read_bytes()).decode()
    data_uri = f"data:{audio_mime_type};base64,{base64_str}"

    # 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization
    url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"
    payload = {
        "model": "qwen-voice-enrollment",  # 不要修改该值
        "input": {
            "action": "create",
            "target_model": target_model,
            "preferred_name": preferred_name,
            "audio": {"data": data_uri}
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, json=payload, headers=headers)
    # {'output': {'target_model': 'qwen3-tts-vc-realtime-2026-01-15', 'voice': 'qwen-tts-vc-guanyu-voice-20260414102610834-3123'}, 'usage': {'count': 1}, 'request_id': '96a6fd16-3423-9bde-bd62-82cae23e32dc'}
    print(resp.json())
    if resp.status_code != 200:
        raise RuntimeError(f"创建 voice 失败: {resp.status_code}, {resp.text}")

    try:
        return resp.json()["output"]["voice"]
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"解析 voice 响应失败: {e}")


def init_dashscope_api_key():
    """
    初始化 dashscope SDK 的 API key
    """
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    # 若没有配置环境变量，请用百炼API Key将下行替换为：dashscope.api_key = "sk-xxx"
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ======= 回调类 =======


class MyCallback(QwenTtsRealtimeCallback):
    """
    自定义 TTS 流式回调
    """

    def __init__(self):
        self.complete_event = threading.Event()
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=8000, output=True
        )

    def on_open(self) -> None:
        print('[TTS] 连接已建立')

    def on_close(self, close_status_code, close_msg) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()
        print(f'[TTS] 连接关闭 code={close_status_code}, msg={close_msg}')

    def on_event(self, response: dict) -> None:
        try:
            event_type = response.get('type', '')
            if event_type == 'session.created':
                print(f'[TTS] 会话开始: {response["session"]["id"]}')
            elif event_type == 'response.audio.delta':
                audio_data = base64.b64decode(response['delta'])
                # 将 24000Hz PCM 转为 8000Hz
                import numpy as np
                import io

                pcm_data = np.frombuffer(audio_data, dtype=np.int16)
                # 使用简单重采样（线性插值）

                def resample_pcm(data, src_rate, tgt_rate):
                    import numpy as np
                    if src_rate == tgt_rate:
                        return data
                    duration = len(data) / src_rate
                    old_indices = np.arange(len(data))
                    new_length = int(len(data) * tgt_rate / src_rate)
                    new_indices = np.linspace(0, len(data) - 1, new_length)
                    new_data = np.interp(
                        new_indices, old_indices, data).astype(np.int16)
                    return new_data
                resampled_pcm = resample_pcm(pcm_data, 24000, 8000)
                self._stream.write(resampled_pcm.tobytes())

            elif event_type == 'response.done':
                print(
                    f'[TTS] 响应完成, Response ID: {qwen_tts_realtime.get_last_response_id()}')
            elif event_type == 'session.finished':
                print('[TTS] 会话结束')
                self.complete_event.set()
        except Exception as e:
            print(f'[Error] 处理回调事件异常: {e}')

    def wait_for_finished(self):
        self.complete_event.wait()


# ======= 主执行逻辑 =======
if __name__ == '__main__':
    init_dashscope_api_key()
    print('[系统] 初始化 Qwen TTS Realtime ...')

    callback = MyCallback()
    qwen_tts_realtime = QwenTtsRealtime(
        model=DEFAULT_TARGET_MODEL,
        callback=callback,
        url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
    )
    qwen_tts_realtime.connect()

    qwen_tts_realtime.update_session(
        voice=create_voice(VOICE_FILE_PATH),  # 将voice参数替换为复刻生成的专属音色
        # voice='qwen-tts-vc-guanyu-voice-20260414111228720-abad',
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode='server_commit'
    )

    # for text_chunk in ''.join(TEXT_TO_SYNTHESIZE):
    for text_chunk in TEXT_TO_SYNTHESIZE:
        print(f'[发送文本]: {text_chunk}')
        qwen_tts_realtime.append_text(text_chunk)
        time.sleep(0.1)

    qwen_tts_realtime.finish()
    callback.wait_for_finished()

    print(f'[Metric] session_id={qwen_tts_realtime.get_session_id()}, '
          f'first_audio_delay={qwen_tts_realtime.get_first_audio_delay()}s')
