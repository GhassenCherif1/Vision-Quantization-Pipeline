FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Pin cupy explicitly BEFORE modelopt pulls in a newer one transitively.
# 13.4.1 is the latest 13.x release before 13.5.0 added CUDA 12.9 support,
# which a 12.5-class driver (550-560 series) can't load (CUDA_ERROR_INVALID_IMAGE).
RUN pip install --no-cache-dir "cupy-cuda12x==13.4.1"
# Pin lightly-train to 0.15.0 — 0.15.1 introduced the D-FINE decoder
# architecture (gateway/LQE refinement layers, distributional bbox head),
# which is NOT compatible with checkpoints trained on the older plain
# 4-output bbox head. Loading those checkpoints against >=0.15.1 fails with
# missing/unexpected state_dict keys and a shape mismatch on dec_bbox_head.
RUN pip install --no-cache-dir \
    "lightly-train[onnx,onnxruntime,onnxslim]==0.15.0" \
    "nvidia-modelopt[onnx]" \
    datasets \
    netron \
    pynvml \
    codecarbon
# Re-pin cupy AFTER modelopt installs, in case modelopt's resolver upgraded it
RUN pip install --no-cache-dir --force-reinstall "cupy-cuda12x==13.4.1"
RUN pip install --no-cache-dir --force-reinstall onnxruntime-gpu
ENV CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
ENV LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/"
EXPOSE 9985
