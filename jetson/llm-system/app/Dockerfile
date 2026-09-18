# Install uv
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt update && \
    apt install -y gcc portaudio19-dev curl

# Use the system Python across both stages
ENV UV_PYTHON_DOWNLOADS=0

# Change the working directory to the `app` directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

# Copy the project into the intermediate image
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

RUN mkdir -p /app/models && \
    curl -L \
      https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx \
      -o /app/models/silero_vad.onnx

## for update pyproject.toml and uv.lock
## docker build --target dev -t localhost/audio-app:dev .
## docker run -d --rm --name dev localhost/audio-app:dev sleep infinity
## docker cp dev:/app/pyproject.toml ./
## docker cp dev:/app/uv.lock ./
#FROM builder as dev
#RUN uv add openai

FROM python:3.12-slim

# Copy the environment, but not the source code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
RUN apt update && \
    apt install -y libportaudio2 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Run the application
CMD ["/app/.venv/bin/app"]
