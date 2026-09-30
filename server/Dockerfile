FROM nvcr.io/nvidia/pytorch:25.09-py3
RUN pip install --no-cache-dir "transformers>=4.56,<5" accelerate "fastapi>=0.115" "uvicorn[standard]" huggingface_hub tokenizers websockets
ENV HF_HOME=/root/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR /app
