import argparse
import os
import numpy as np
from datasets import load_dataset
from torchvision import transforms

def export_calibration_tensors(dataset_name, split, size, out_path, size_dims=(640,640)):
    print(f"📦 Streaming '{split}' split from '{dataset_name}' dataset...")
    
    # Generic object detection preprocess configuration
    calib_transform = transforms.Compose([
        transforms.Resize(size_dims),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    images = []
    for i, sample in enumerate(dataset):
        if i >= size:
            break
        # Grab image object, normalize shape channels, and extend dimensions
        img = sample["image"].convert("RGB")
        tensor_img = calib_transform(img).unsqueeze(0)
        images.append(tensor_img.numpy())
        
    calib_data = np.concatenate(images, axis=0)
    
    # Ensure directory path structures exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, calib_data)
    print(f"🎉 Success! Extracted data shape: {calib_data.shape} saved to -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache clean validation subsets.")
    parser.add_argument("--dataset", default="detection-datasets/coco", help="HuggingFace dataset path")
    parser.add_argument("--split", default="val", help="Dataset target partition")
    parser.add_argument("--size", type=int, default=100, help="Total sample count")
    parser.add_argument("--output", required=True, help="Destination file path (.npy)")
    parser.add_argument("--resolution", type=int, default=640, help="Input model square dimension size")
    args = parser.parse_args()

    export_calibration_tensors(
        dataset_name=args.dataset,
        split=args.split,
        size=args.size,
        out_path=args.output,
        size_dims=(args.resolution, args.resolution)
    )