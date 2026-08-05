import argparse
import os
import sys
import time
import netron

def start_netron_server(model_path, port=9985):
    # 1. Quick sanity check on the file path
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        sys.exit(1)
        
    print(f"🔍 Initializing Netron Visualizer for: {model_path}")
    
    # 2. Clear out any legacy ghost instances running on this port
    try:
        netron.stop()
    except Exception:
        pass

    # 3. Start the server exposed to the Docker bridge network
    print("\n------------------------------------------------------------")
    print(f"🔗 NETRON SERVER IS ALIVE!")
    print(f"👉 Open your host browser and visit: http://localhost:{port} 👈")
    print("   Press CTRL+C in this terminal to shutdown the server.")
    print("------------------------------------------------------------\n")
    
    try:
        # host="0.0.0.0" is critical for Docker container port mapping
        netron.start(model_path, address=("0.0.0.0", port))
        # Keep the process running while you browse the layers
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping Netron server cleanly...")
        netron.stop()
        print("👋 Netron offline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize any ONNX graph using Netron inside Docker.")
    parser.add_argument("--model", required=True, help="Path to the ONNX file (.onnx)")
    parser.add_argument("--port", type=int, default=9985, help="Network port to bind to")
    args = parser.parse_args()

    start_netron_server(args.model, args.port)