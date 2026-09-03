FROM pytorch/pytorch:2.6.0-cuda12.1-cudnn8-runtime

# Install system dependencies (optional, for GGUF conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the script and default config
WORKDIR /app
COPY heretic_v3.py .
COPY config.example.yaml /app/config.example.yaml
# Users can mount their own config.yaml at runtime

# Entry point
ENTRYPOINT ["python", "heretic_v3.py"]
CMD ["--config", "config.yaml"]
