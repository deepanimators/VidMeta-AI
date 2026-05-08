FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIDMETA_MAX_UPLOAD_MB=2048

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY vidmeta ./vidmeta
COPY api ./api
COPY README.md ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["vidmeta", "serve", "--host", "0.0.0.0", "--port", "8000"]
