import json
import lightly_train

with open("coco/class_names.json") as f:
    names = {int(k): v for k, v in json.load(f).items()}

lightly_train.train_object_detection(
    out="out/dinov3_vitb16_ltdetr_coco",
    model="dinov3/vitb16-ltdetr",
    model_args={"backbone_freeze": True},
    data={
        "path": "coco",
        "train": "images/train",
        "val": "images/val",
        "names": names,
    },
    steps=90_000,
    batch_size=8,
    devices="auto",
    resume_interrupted=True,
)
