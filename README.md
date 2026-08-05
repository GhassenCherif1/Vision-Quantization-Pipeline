# Edge Vision Transformer Quantization & Profiling Pipeline

An automated model optimization framework built to prepare, flatten, compress, and profile Vision Transformer architectures (such as ViT/LT-DETR) before edge deployment.

The pipeline uses NVIDIA's **ModelOpt** to generate quantized ONNX graphs across multiple precision strategies (`INT8`, `INT4`) and leverages **CodeCarbon** alongside **NVML** to map host resource consumption (VRAM, Power, Runtime, and Carbon footprint) sequentially without memory contamination.

---

## 🐳 Getting Started (Docker)

### 1. Clone / enter the repo

```bash
cd ~/vit-quant-pipeline
```

### 2. Build the image

```bash
sudo docker build -t vit-quantizer:latest .
```

### 3. Run the container

```bash
sudo docker run --gpus all -it --rm \
  -p 9986:9985 \
  -v $(pwd):/workspace \
  vit-quantizer:latest
```

- `--gpus all` — passes the host GPU(s) through to the container (required for PyTorch/cupy/onnxruntime-gpu/ModelOpt).
- `-p 9986:9985` — maps container port `9985` (used by `visualize_netron.py`) to host port `9986`. Once the container is running, open `visualize_netron.py` on `--port 9985` inside the container, and access it at `http://localhost:9986` on your host.
- `-v $(pwd):/workspace` — mounts the whole repo into `/workspace` in the container, so `data/`, `models/`, and `scripts/` are shared and any outputs (quantized models, `matrix_results.csv`, `emissions.csv`) persist back to the host after the container exits.
- `-it --rm` — interactive shell, and the container is removed on exit (your files are safe since they live on the mounted host volume, not inside the container).

This drops you into a shell inside the container at `/workspace`, where you can run the pipeline scripts below.

---

## 🚀 Pipeline Architecture Overview

The framework separates structural graph operations, data prep, and matrix strategy management from the execution of individual optimization passes. This decoupled layout provides operational fault tolerance on consumer GPUs with limited VRAM.

```text
[1. Raw Model] ──► flatten_and_view.py ──► [2. Flattened Model]
                        │
[3. COCO Images] ──► prepare_calibration.py ───────┼──► run_grid_search.py (Orchestrator)
                        │                           │
                        ▼                           ├──► Subprocess 1: quantize_with_telemetry.py
               visualize_netron.py                 ├──► Subprocess 2: quantize_with_telemetry.py (OOM-Safe)
                                                    └──► Subprocess 3: ...
```

### Key Highlights

* **Sequential Process Isolation:** `run_grid_search.py` launches `quantize_with_telemetry.py` as a detached OS child subprocess using `subprocess.run`. If a configuration triggers a `CUDA Out of Memory (OOM)` exception, the master orchestrator catches the failure, logs it, and safely proceeds.
* **Flushed VRAM Cache:** Because child processes fully terminate between runs, the operating system and NVIDIA driver instantly reclaim 100% of the allocated VRAM. This prevents memory bloat across back-to-back iterations.
* **Atomic Telemetry Logging:** Metrics are appended to a centralized database file row-by-row instantly after each execution window closes, safeguarding tracking records against mid-sweep system crashes.

---

## 📂 Project Structure

```text
.
├── Dockerfile                      # Containerized build environment
├── emissions.csv                   # Global CodeCarbon trace output
├── data/
│   └── processed_calib/
│       └── coco_val_100.npy        # Pre-built 100-image calibration tensor
├── models/
│   ├── raw/
│   │   └── vitt16-ltdetr-coco.onnx # Original exported model graph
│   ├── flattened/
│   │   ├── vits16-ltdetr-coco-flat.onnx # Flattened Small-backbone variant
│   │   └── vitt16-ltdetr-coco-flat.onnx # Flattened Tiny-backbone variant
│   └── quantized/
│       ├── matrix_results.csv      # Centralized database of run summaries (Auto-generated)
│       └── *.onnx                  # Generated optimization candidates (Ignored by Git)
└── scripts/
    ├── flatten_and_view.py         # Step 1: Prepares the raw graph by flattening dynamic shapes
    ├── prepare_calibration.py      # Step 2: Processes and packages calibration image matrices
    ├── quantize_with_telemetry.py  # Step 3: Core ModelOpt wrapper script with active telemetry
    ├── run_grid_search.py          # Step 4: Master grid sweep orchestration runner
    └── visualize_netron.py         # Diagnostic: Spins up a local Netron graph visualization server
```

---

# 💻 Script Directory & Execution Guide

All commands below are run **inside the container**, from `/workspace` (after following the Getting Started steps above).

## 1. Graph Preparation (`flatten_and_view.py`)

Flattens dynamic dimensions, executes an initial ONNX shape inference pass, and prepares the model structure for quantization tool inputs.

```bash
python scripts/flatten_and_view.py \
  --input models/raw/vitt16-ltdetr-coco.onnx \
  --output models/flattened/vitt16-ltdetr-coco-flat.onnx
```

---

## 2. Calibration Sampling (`prepare_calibration.py`)

Loads a representative validation image subset (e.g., COCO), resizes/normalizes tensors to match input configurations, and packs them into a lightweight binary NumPy array.

```bash
python scripts/prepare_calibration.py \
  --image_dir /path/to/coco/val2017 \
  --output data/processed_calib/coco_val_100.npy \
  --num_images 100
```

---

## 3. Individual Quantization Pass (`quantize_with_telemetry.py`)

Runs a single, targeted quantization pass on the model while measuring VRAM, execution time, power draw, and carbon efficiency.

> **Note for Limited GPU:** Always use `--calibrate_per_node` and `--simplify` on large models to process the graph layer-by-layer and avoid hardware OOM crashes.

```bash
python scripts/quantize_with_telemetry.py \
  --input models/flattened/vitt16-ltdetr-coco-flat.onnx \
  --calib_file data/processed_calib/coco_val_100.npy \
  --output models/quantized/vitt16_int8_entropy_fp16_safeMHA.onnx \
  --mode int8 \
  --calib_method entropy \
  --high_precision_dtype fp16 \
  --calibrate_per_node \
  --simplify
```

---

## 4. Automated Matrix Search (`run_grid_search.py`)

The master orchestrator. Automatically loops over 24 distinct precision permutations (INT8, INT4, Selective Attention Isolation, and Precision Fallbacks) using sequential subprocesses.

### Default Execution

```bash
python scripts/run_grid_search.py
```

### Custom Grid Sweep

To run a custom grid sweep without modifying code:

```bash
python scripts/run_grid_search.py \
  --base_model models/flattened/vits16-ltdetr-coco-flat.onnx \
  --calib_file data/processed_calib/coco_val_100.npy \
  --output_dir models/quantized
```

---

## 5. Graph Inspection (`visualize_netron.py`)

Spins up a local Netron web server to inspect the structural changes, node placements, and Q/DQ (Quantize/Dequantize) layer configurations in your generated ONNX graphs.

```bash
python scripts/visualize_netron.py \
  --model models/quantized/vitt16_int8_entropy_fp16_safeMHA.onnx \
  --port 9985
```

Since the container was started with `-p 9986:9985`, view the visualizer on your host at `http://localhost:9986`.