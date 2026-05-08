FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIDMETA_MAX_UPLOAD_MB=2048 \
    VIDMETA_MAX_MESSAGE_MB=2048

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY vidmeta ./vidmeta
COPY app.py README.md ./
COPY .streamlit ./.streamlit

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

EXPOSE 8501

CMD ["vidmeta", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
