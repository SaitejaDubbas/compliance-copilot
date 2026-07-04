FROM python:3.11-slim

# PyTorch's and FAISS's CPU backends link against OpenMP at runtime; the slim
# base image doesn't include it by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces' Docker SDK requires the container to run as a non-root
# user (uid 1000).
RUN useradd -m -u 1000 appuser
USER appuser
WORKDIR /home/appuser/app
ENV PATH="/home/appuser/.local/bin:${PATH}"

# The default PyPI torch wheel bundles CUDA libraries this CPU-only deployment
# never uses; installing from the official CPU index keeps the image small.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/labels.json ./data/labels.json

ENV PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
