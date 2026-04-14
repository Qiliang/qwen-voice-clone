# --- 第一阶段：构建阶段 ---
    FROM astral/uv:python3.11-trixie AS builder
    WORKDIR /app
    ENV UV_HTTP_TIMEOUT=300
    # 仅复制依赖文件（利用 Docker 缓存层）
    # 如果 uv.lock 或 pyproject.toml 没变，这一层会被缓存
    COPY pyproject.toml uv.lock ./
    RUN uv sync --frozen --no-cache --no-install-project --no-dev
    COPY . .
    
    # --- 第二阶段：运行阶段 ---
    FROM astral/uv:python3.11-trixie
    WORKDIR /app
    ENV TZ=Asia/Shanghai
    COPY --from=builder /app/.venv /app/.venv
    COPY . .
    # 启动fastapi: uv run fastapi run
    CMD [ "uv", "run", "fastapi","run","--workers","4","--port","8080"]
    