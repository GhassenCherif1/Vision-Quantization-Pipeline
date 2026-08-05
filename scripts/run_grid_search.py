# run_grid_search.py
import argparse
import os
import csv
import subprocess
import itertools

# ==============================================================================
# CONFIGURATION SEARCH SPACE MATRIX
# ==============================================================================
QUANT_MODES = ["int8", "int4"]
HIGH_PRECISION_DTYPES = ["fp16", "fp32"]
DISABLE_MHA_FLAGS = [True, False]  # True keeps MHA in high precision, False quantizes it

def get_calib_methods(mode):
    """Auto-mapping calibration methods based on chosen mode rules."""
    if mode == "int4":
        return ["awq_clip", "awq_lite", "awq_full", "rtn_dq"]
    return ["entropy", "max"]

# ==============================================================================
# CSV STORAGE TRANS-LOG TRANSACTION
# ==============================================================================
def initialize_csv(output_dir, results_csv):
    """Creates the master results log file with headers if it doesn't exist yet."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(results_csv):
        headers = [
            "Model_File", "Quant_Mode", "Calib_Method", "High_Precision_Dtype", 
            "MHA_Status", "Status", "Runtime_Sec", "Peak_VRAM_MB", 
            "Avg_GPU_Watts", "Avg_CPU_Watts", "Avg_RAM_Watts", 
            "Total_Energy_kWh", "Carbon_Produced_kg", "Grid_Intensity"
        ]
        with open(results_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_csv(results_csv, row_data):
    """Atomically appends one run's metric row straight to disk."""
    with open(results_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

# ==============================================================================
# MAIN EXECUTION ENGINE LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Automated Grid Optimization Sweep Orchestrator")
    parser.add_argument("--base_model", required=True, 
                        help="Path to input flat ONNX model")
    parser.add_argument("--calib_file", default="data/processed_calib/coco_val_100.npy", 
                        help="Path to validation calibration dataset (.npy)")
    parser.add_argument("--output_dir", default="models/quantized", 
                        help="Target output directory for compiled variants")
    args = parser.parse_args()

    # Derived file paths based on inputs
    results_csv = os.path.join(args.output_dir, "matrix_results.csv")
    initialize_csv(args.output_dir, results_csv)
    
    # Generate static grid combinations
    combinations = list(itertools.product(QUANT_MODES, HIGH_PRECISION_DTYPES, DISABLE_MHA_FLAGS))
    
    print(f"🎬 Starting Grid Optimization Sweep.")
    print(f"• Input Graph : {args.base_model}")
    print(f"• Calib Matrix: {args.calib_file}")
    print(f"• Output Port : {args.output_dir}")
    print(f"• Matrix Registry: {results_csv}")
    print(f"• Total Strategy Groups to run: {len(combinations) * 2}\n")

    for mode, high_precision_dtype, disable_mha in combinations:
        methods = get_calib_methods(mode)
        
        for method in methods:
            mha_tag = "safeMHA" if disable_mha else "quantMHA"
            model_name = f"convnext_base_{mode}_{method}_{high_precision_dtype}_{mha_tag}"
            output_model_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            
            print(f"🔄 Spawning Child Subprocess -> Variant Strategy: {model_name}...")

            # Assemble CLI arguments to execute your modular python component
            cmd = [
                "python", "scripts/quantize_with_telemetry.py",
                "--input", args.base_model,
                "--output", output_model_path,
                "--calib_file", args.calib_file,
                "--mode", mode,
                "--calib_method", method,
                "--high_precision_dtype", high_precision_dtype,
                "--country", "DEU"
            ]
            
            if disable_mha:
                cmd.append("--disable_mha_qdq")
                
            # If tracking int4, automatically apply standard block sizes
            if mode == "int4":
                cmd.extend(["--block_size", "128"])

            # Execute run as an isolated subprocess block to catch crashes safely
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✅ Subprocess closed cleanly: {model_name} written.")
                
            except subprocess.CalledProcessError:
                print(f"❌ Subprocess Crashed (Likely OOM): {model_name}")
                
                # Write an explicit failure log line item into your telemetry metrics file
                failure_row = [
                    f"{model_name}.onnx", mode, method, high_precision_dtype, 
                    mha_tag, "FAILED_OR_OOM", "0.0", "0.0", 
                    "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"
                ]
                append_to_csv(results_csv, failure_row)

if __name__ == "__main__":
    main()