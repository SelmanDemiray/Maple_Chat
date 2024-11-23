# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash myuser

# Set up environment variables
ENV HOME=/home/myuser \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/home/myuser/.cache/huggingface \
    HF_HOME=/home/myuser/.cache/huggingface

# Create cache directories and set permissions
RUN mkdir -p ${HF_HOME} && \
    chown -R myuser:myuser ${HOME}

# Ensure python3 is the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Switch to non-root user
USER myuser

# Copy requirements file
COPY --chown=myuser:myuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=myuser:myuser . .

# Pre-download model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model_name = 'lmsys/vicuna-7b-v1.5'; \
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)"

# Expose port
EXPOSE 8007

# Start command
CMD ["python", "main.py"]