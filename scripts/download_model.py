import torch
import lightly_train

model = lightly_train.load_model("dinov3/vits16-eomt-panoptic-coco",device="cpu")
model.export_onnx(
    out="segmentation_panoptic.onnx",
    # precision="fp16", # Export model with FP16 weights for smaller size and faster inference.
)
