import argparse
import time
import netron
from onnxruntime.transformers import optimizer

def optimize_and_flatten(input_path, output_path):
    print(f"🚀 Loading model into ONNX Runtime Transformer Optimizer: {input_path}")
    
    optimized_model = optimizer.optimize_model(
        input_path,
        model_type='bert',  # Fuses transformer layers
        num_heads=16,
        hidden_size=0,
        opt_level=1,        # Level 1 performs basic flattening & constant folding
        only_onnxruntime=True,
        use_gpu=False       # Forces standard CPU node signatures
    )
    
    optimized_model.save_model_to_file(output_path)
    print(f"🎉 Flattened model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw ONNX model")
    parser.add_argument("--output", required=True, help="Path to save flat ONNX model")
    parser.add_argument("--view", action="store_true", help="Launch Netron visualizer after processing")
    args = parser.parse_args()

    # Run the flattening process
    optimize_and_flatten(args.input, args.output)

    # Netron Debugging Layer
    if args.view:
        print("\n------------------------------------------------------------")
        print(f"🔗 NETRON SERVER STARTED. Open your host browser and visit:")
        print(f"    👉 http://localhost:9985 👈")
        print("    Press CTRL+C in this terminal to stop the server.")
        print("------------------------------------------------------------\n")
        
        try:
            netron.start(args.output, address=("0.0.0.0", 9985))
            # Keep the script alive while you look at the graph
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping Netron visualizer...")
            netron.stop()