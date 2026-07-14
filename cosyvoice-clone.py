import os
import shutil
import subprocess
import tempfile
import time
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService, SpeechSynthesizer
from dashscope.audio.tts_v2.speech_synthesizer import AudioFormat

from rustfs import upload_file

OUTPUT_SAMPLE_RATE = 8000


def main(text="恭喜，已成功复刻并合成了属于自己的声音！", voice_id = None):
    # 1. 环境准备
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    print(dashscope.api_key)
    if not dashscope.api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

    # 以下为北京地域WebSocket url，若使用新加坡地域的模型，需将url替换为：wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference
    dashscope.base_websocket_api_url='wss://dashscope.aliyuncs.com/api-ws/v1/inference'
    # 以下为北京地域HTTP url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
    dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


    # 2. 定义复刻参数
    TARGET_MODEL = "cosyvoice-v3.5-flash" 
    # 为音色起一个有意义的前缀
    VOICE_PREFIX = "chengna" # 仅允许数字和小写字母，小于十个字符

    if not voice_id:
        # 本地音频路径：先上传到 rustfs，再使用返回的公网地址
        AUDIO_PATH = "/Users/xiaoql/Downloads/合力咨询_程娜_1_clip1.wav"
        AUDIO_URL = upload_file(AUDIO_PATH)
        print(f"Uploaded to rustfs: {AUDIO_URL}")

        # 3. 创建音色 (异步任务)
        print("--- Step 1: Creating voice enrollment ---")
        service = VoiceEnrollmentService()
        try:
            voice_id = service.create_voice(
                target_model=TARGET_MODEL,
                prefix=VOICE_PREFIX,
                url=AUDIO_URL,
                max_prompt_audio_length=30,
            )
            print(f"Voice enrollment submitted successfully. Request ID: {service.get_last_request_id()}")
            print(f"Generated Voice ID: {voice_id}")
        except Exception as e:
            print(f"Error during voice creation: {e}")
            raise e
        # 4. 轮询查询音色状态
        print("\n--- Step 2: Polling for voice status ---")
        max_attempts = 30
        poll_interval = 10 # 秒
        for attempt in range(max_attempts):
            try:
                voice_info = service.query_voice(voice_id=voice_id)
                status = voice_info.get("status")
                print(f"Attempt {attempt + 1}/{max_attempts}: Voice status is '{status}'")
                
                if status == "OK":
                    print("Voice is ready for synthesis.")
                    break
                elif status == "UNDEPLOYED":
                    print(f"Voice processing failed with status: {status}. Please check audio quality or contact support.")
                    raise RuntimeError(f"Voice processing failed with status: {status}")
                # 对于 "DEPLOYING" 等中间状态，继续等待
                time.sleep(poll_interval)
            except Exception as e:
                print(f"Error during status polling: {e}")
                time.sleep(poll_interval)
        else:
            print("Polling timed out. The voice is not ready after several attempts.")
            raise RuntimeError("Polling timed out. The voice is not ready after several attempts.")

    # 5. 使用复刻音色进行语音合成
    print("\n--- Step 3: Synthesizing speech with the voice ---")
    print(f"Using voice_id: {voice_id}")
    try:
        synthesizer = SpeechSynthesizer(
            model=TARGET_MODEL, 
            voice=voice_id,
            format=AudioFormat.WAV_8000HZ_MONO_16BIT
            )
        text_to_synthesize = text
        
        # call()方法返回二进制音频数据
        audio_data = synthesizer.call(text_to_synthesize)
        request_id = synthesizer.get_last_request_id()
        if not audio_data:
            raise RuntimeError(
                f"Speech synthesis returned empty audio. Request ID: {request_id}. "
                "Usually means voice_id is invalid / not OK, or model/voice mismatch (error 418)."
            )
        print(f"Speech synthesis successful. Request ID: {request_id}")

        # 6. 保存音频文件
        output_file = "my_custom_voice_output.wav"

        with open(output_file, "wb") as f:
            f.write(audio_data)
        print(
            f"Audio saved to {output_file} (sample_rate={OUTPUT_SAMPLE_RATE}, mono)"
        )

    except Exception as e:
        print(f"Error during speech synthesis: {e}")
        raise



if __name__ == "__main__":
    text="""
嗯，可以，我们客服系统的话，功能比较全面，像在线通话工单这些功能都包括的。您这边主要是哪一个模块的需求呢？嗯，可以通话的话，是否在智能接听那个需求呢？
    """
    # 传 None 会走上传 + 复刻；有可用 voice_id 时可直接传入跳过复刻
    # voice_id = "cosyvoice-v3.5-plus-chengna-9b3ca8ec0c214c46af9d8031de3d5cad"
    voice_id = "cosyvoice-v3.5-flash-chengna-36783aa020fb4fb399a041f943870f3b"
    # voice_id = None
    main(text, voice_id)