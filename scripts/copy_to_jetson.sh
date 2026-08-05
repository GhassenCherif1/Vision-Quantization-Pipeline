#!/bin/bash

SOURCE_DIR="/home/ghassencherif/vit-quant-pipeline/models/quantized"
DEST_USER="ghassen"
DEST_HOST="141.79.68.41"
DEST_DIR="/home/ghassen/inference_pipeline/models/onnx_models"

scp -r "$SOURCE_DIR" "${DEST_USER}@${DEST_HOST}:${DEST_DIR}"
