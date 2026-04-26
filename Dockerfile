# --- 第一阶段：构建阶段 ---
    FROM astral/uv:python3.11-trixie AS builder
    WORKDIR /app
    ENV UV_HTTP_TIMEOUT=600
    # ffmpeg 是 yt-dlp / Demucs / librosa 的硬依赖
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && rm -rf /var/lib/apt/lists/*
    # 仅复制依赖文件（利用 Docker 缓存层）
    # 如果 uv.lock 或 pyproject.toml 没变，这一层会被缓存
    COPY pyproject.toml uv.lock ./
    RUN uv sync --frozen --no-cache --no-install-project --no-dev
    COPY . .
    # 预下载 Demucs htdemucs 与 SpeechBrain ECAPA-TDNN 权重，避免首次请求超时。
    # 关键：先 mkdir 兜底，保证 pretrained_models 目录一定存在；
    # 即便下载失败（网络抖动），stage-1 的 COPY --from=builder 也不会因目录不存在而 fail。
    RUN mkdir -p /app/pretrained_models/spkrec-ecapa-voxceleb \
        && (uv run python -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
            || echo "[WARN] demucs htdemucs 预下载失败，将在运行时按需下载") \
        && (uv run python -c "from speechbrain.inference.speaker import EncoderClassifier; from speechbrain.utils.fetching import LocalStrategy; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb', local_strategy=LocalStrategy.COPY)" \
            || echo "[WARN] speechbrain ECAPA-TDNN 预下载失败，将在运行时按需下载")

    # --- 第二阶段：运行阶段 ---
    FROM astral/uv:python3.11-trixie
    WORKDIR /app
    ENV TZ=Asia/Shanghai
    RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
        && rm -rf /var/lib/apt/lists/*
    COPY --from=builder /app/.venv /app/.venv
    COPY --from=builder /root/.cache /root/.cache
    COPY . .
    # 把 builder 阶段缓存的 SpeechBrain 模型一并带过来。
    # builder 阶段已用 mkdir -p 兜底，这里 COPY 不会因目录缺失而 fail。
    COPY --from=builder /app/pretrained_models /app/pretrained_models
    # 启动fastapi: uv run fastapi run
    CMD [ "uv", "run", "fastapi","run","--workers","4","--port","8080"]
