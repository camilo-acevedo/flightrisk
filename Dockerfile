FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY flightrisk ./flightrisk

RUN pip install --upgrade pip wheel setuptools \
    && pip install --prefix=/install .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLIGHTRISK_LOG_LEVEL=INFO \
    MPLBACKEND=Agg \
    PORT=8000

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system flightrisk \
    && useradd --system --gid flightrisk --home /home/flightrisk --create-home flightrisk

WORKDIR /app

COPY --from=builder /install /usr/local
COPY flightrisk ./flightrisk
COPY pyproject.toml README.md ./

RUN mkdir -p /app/data /app/reports /app/mlruns \
    && chown -R flightrisk:flightrisk /app

USER flightrisk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --silent --fail http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn flightrisk.serving.api:app --host 0.0.0.0 --port ${PORT}"]
