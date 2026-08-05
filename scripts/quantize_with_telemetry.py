# scripts/03_quantize_with_telemetry.py
import argparse
import time
import threading
import numpy as np
import pynvml
from codecarbon import OfflineEmissionsTracker
from modelopt.onnx.quantization import quantize

try:
    import cupy
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    print("✅ CuPy memory pool flushed")
except Exception as e:
    print(f"ℹ️ CuPy pool flush skipped: {e}")

telemetry = {"peak_vram_bytes": 0, "power_samples": [], "stop": False}

def monitor_gtx_hardware(gpu_index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    while not telemetry["stop"]:
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if mem_info.used > telemetry["peak_vram_bytes"]:
                telemetry["peak_vram_bytes"] = mem_info.used
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            telemetry["power_samples"].append(power)
        except Exception:
            pass
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced ModelOpt Quantization Wrapper with Hardware Tracking")
    # Core Paths
    parser.add_argument("--input", required=True, help="Flattened input ONNX model")
    parser.add_argument("--output", required=True, help="Output Quantized ONNX file path")
    parser.add_argument("--calib_file", required=True, help="Path to prepared calibration .npy file")
    
    # Precision Mode Knobs
    parser.add_argument("--mode", choices=["int8", "fp8", "int4"], default="int8", help="Target quantization format")
    parser.add_argument("--calib_method", default=None, 
                        help="Calibration scale algorithm. int8/fp8: {'entropy', 'max'}. int4: {'awq_clip', 'awq_lite', 'awq_full', 'rtn_dq'}")
    parser.add_argument("--high_precision_dtype", choices=["fp16", "fp32", "bf16"], default="fp16", help="Fallback data type for weights/activations")
    
    # Advanced Quantization Struct Knobs
    parser.add_argument("--block_size", type=int, default=None, help="Block size parameter specifically for int4 formats")
    parser.add_argument("--use_zero_point", action="store_true", help="Activate zero-point based asymmetric quantization")
    parser.add_argument("--dq_only", action="store_true", help="Only add Dequantize (DQ) nodes to the model layout")
    
    # Transformer MHA Specialized Precision Knobs
    parser.add_argument("--mha_accumulation_dtype", choices=["fp16", "fp32"], default="fp16", help="Accumulation precision inside MHA blocks")
    parser.add_argument("--disable_mha_qdq", action="store_true", help="Skip inserting Q/DQ layers inside multi-head attention MatMuls")
    
    # Layer Selective Isolation Knobs
    parser.add_argument("--op_types_to_quantize", nargs="+", default=None, help="OP nodes targeted for QDQ insertion (None = all supported)")
    parser.add_argument("--op_types_to_exclude", nargs="+", default=None, help="Force specific OP signatures to remain unquantized")
    parser.add_argument("--nodes_to_exclude", nargs="+", default=None, help="Explicit regex or names of individual layers to skip")
    
    # Struct optimization flags
    parser.add_argument("--simplify", action="store_true", help="Execute an internal structural simplification graph pass")
    parser.add_argument("--calibrate_per_node", action="store_true", help="Calibrate scales node-by-node (Saves VRAM on host card)")
    parser.add_argument("--use_external_data_format", action="store_true", help="Split weights into an external binary file (Required if model >2GB)")
    
    # Telemetry configurations
    parser.add_argument("--country", default="DEU")
    args = parser.parse_args()

    # Dynamic fallback defaults based on NVIDIA documentation rules
    if args.calib_method is None:
        args.calib_method = "awq_clip" if args.mode == "int4" else "entropy"

    # STEP 1: Fast direct memory initialization from cache file
    print(f"📁 Loading cached calibration file: {args.calib_file}")
    calib_data = np.load(args.calib_file)

    # STEP 2: Initialize trackers
    tracker = OfflineEmissionsTracker(
        country_iso_code=args.country,
        log_level="error",
        save_to_file=True,
        output_dir="/workspace"
    )
    monitor_thread = threading.Thread(target=monitor_gtx_hardware, args=(0,))
    
    start_time = time.time()
    monitor_thread.start()
    tracker.start()

    # STEP 3: Run target optimization with comprehensive argument integration
    print(f"💎 Executing ModelOpt Quantization Pass (Mode: {args.mode}, Method: {args.calib_method})...")
    try:
        quantize(
            onnx_path=args.input,
            quantize_mode=args.mode,
            calibration_data=calib_data,
            calibration_method=args.calib_method,
            calibration_eps=["cuda:0","cpu"],
            high_precision_dtype=args.high_precision_dtype,
            block_size=args.block_size,
            use_zero_point=args.use_zero_point,
            dq_only=args.dq_only,
            mha_accumulation_dtype=args.mha_accumulation_dtype,
            disable_mha_qdq=args.disable_mha_qdq,
            output_path=args.output,
            op_types_to_quantize=args.op_types_to_quantize,
            op_types_to_exclude=args.op_types_to_exclude,
            nodes_to_exclude=args.nodes_to_exclude,
            simplify=args.simplify,
            calibrate_per_node=args.calibrate_per_node,
            use_external_data_format=args.use_external_data_format,
            disable_autocast=False,
        )
    finally:
        execution_time = time.time() - start_time
        emissions_kg = tracker.stop()
        telemetry["stop"] = True
        monitor_thread.join()

    # ==============================================================================
    # STEP 4: Reporting & Appending into Matrix Database
    # ==============================================================================
    cc_data = tracker.final_emissions_data
    peak_vram_mb = telemetry["peak_vram_bytes"] / (1024 ** 2)
    avg_gpu_power = sum(telemetry["power_samples"]) / max(len(telemetry["power_samples"]), 1)
    calculated_intensity = (cc_data.emissions * 1000) / cc_data.energy_consumed if cc_data.energy_consumed > 0 else 0.0

    # DYNAMICALLY DETECT TOTAL VRAM
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assumes GPU 0
        total_vram_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        total_vram_mb = total_vram_bytes / (1024 ** 2)
    except Exception:
        total_vram_mb = 4096.0  # Fallback if NVML fails

    mha_status = "safeMHA" if args.disable_mha_qdq else "quantMHA"

    print("\n===========================================")
    print("📈 HOST QUANTIZATION RESOURCE METRICS")
    print("===========================================")
    print(f"• Wall Runtime       : {cc_data.duration:.2f} seconds")
    print(f"• Peak VRAM Usage    : {peak_vram_mb:.2f} MB / {total_vram_mb:.0f} MB")
    print("\n⚡ HARDWARE POWER BREAKDOWN (CodeCarbon)")
    print(f"• Avg GPU Power Draw : {avg_gpu_power:.2f} Watts")
    print(f"• Avg CPU Power Draw : {cc_data.cpu_power:.2f} Watts")
    print(f"• Avg RAM Power Draw : {cc_data.ram_power:.2f} Watts")
    print(f"• Total Energy Spent : {cc_data.energy_consumed:.5f} kWh")
    print("\n🌍 ENVIRONMENTAL SUSTAINABILITY METRICS")
    print(f"• Grid Carbon Intensity : {calculated_intensity:.2f} gCO2/kWh")
    print(f"• Net Carbon Produced   : {cc_data.emissions:.6f} kg of CO2")
    print("===========================================\n")

    # ATOMIC TRANSACTION APPEND DIRECTLY TO THE SWEEP MATRIX FILE
    import os
    import csv
    matrix_csv_path = "/workspace/models/quantized/matrix_results.csv"
    
    if os.path.exists(matrix_csv_path):
        success_row = [
            os.path.basename(args.output), args.mode, args.calib_method, args.high_precision_dtype,
            mha_status, "SUCCESS", f"{cc_data.duration:.2f}", f"{peak_vram_mb:.2f}",
            f"{avg_gpu_power:.2f}", f"{cc_data.cpu_power:.2f}", f"{cc_data.ram_power:.2f}",
            f"{cc_data.energy_consumed:.5f}", f"{cc_data.emissions:.6f}", f"{calculated_intensity:.2f}"
        ]
        with open(matrix_csv_path, mode="a", newline="") as f:
            csv.writer(f).writerow(success_row)
