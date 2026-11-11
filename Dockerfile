# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.8.20-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV HUPKES_BIN=/opt/venv/bin
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
WORKDIR /work
CMD ["python", "train.py"]
