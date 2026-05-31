# ELANG — Hugging Face Spaces image (Docker SDK)
# Lightweight stack for the free CPU tier: vehicle detection, tracking,
# violation heatmap + officer optimizer, E-TLE export, and citizen-report CRM.
# PaddleOCR/ANPR is intentionally left out here to keep the build small.

FROM python:3.10-slim

# System libraries OpenCV needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as UID 1000 — create a matching user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Python dependencies (slim hosted stack)
COPY --chown=user requirements-hf.txt ./
RUN pip install --no-cache-dir --user -r requirements-hf.txt

# Application code
COPY --chown=user . .

# Writable cache locations for model weights / configs
ENV YOLO_CONFIG_DIR=/home/user/app/.cache \
    MPLCONFIGDIR=/home/user/app/.cache \
    HF_HOME=/home/user/app/.cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
