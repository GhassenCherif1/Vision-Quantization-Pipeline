"""
Convert COCO 2017 instance annotations (JSON) to YOLO-format label files
(.txt per image, normalized class_id cx cy w h) for LightlyTrain LT-DETR.

Usage:
    python coco_to_yolo.py \
        --ann /path/to/annotations/instances_train2017.json \
        --img-dir /path/to/train2017 \
        --out-labels /path/to/coco/labels/train \
        --out-names-json /path/to/coco/class_names.json   # only needs to be written once (train pass)

Run once for train, once for val (pointing at the val json/img dir/out dir).
The class_names.json produced on the train pass is the canonical id->name
mapping to paste into your `names={...}` dict when calling
lightly_train.train_object_detection().
"""

import argparse
import json
import os
from collections import defaultdict


def convert(ann_path: str, img_dir: str, out_labels_dir: str, out_names_json: str | None):
    with open(ann_path, "r") as f:
        coco = json.load(f)

    os.makedirs(out_labels_dir, exist_ok=True)

    # --- 1. Build a contiguous 0..N-1 class id mapping from COCO's sparse category ids ---
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    coco_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(categories)}
    yolo_id_to_name = {i: cat["name"] for i, cat in enumerate(categories)}

    if out_names_json:
        with open(out_names_json, "w") as f:
            json.dump(yolo_id_to_name, f, indent=2)
        print(f"Wrote {len(yolo_id_to_name)}-class id->name mapping to {out_names_json}")

    # --- 2. Index images by id ---
    images = {img["id"]: img for img in coco["images"]}

    # --- 3. Group annotations by image id ---
    anns_by_image = defaultdict(list)
    skipped_degenerate = 0
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            # LT-DETR / YOLO-format training typically excludes crowd regions;
            # drop them rather than treating them as normal boxes.
            continue
        x, y, w, h = ann["bbox"]  # COCO: absolute pixel, top-left x,y + width,height
        if w <= 0 or h <= 0:
            skipped_degenerate += 1
            continue
        anns_by_image[ann["image_id"]].append((ann["category_id"], x, y, w, h))

    # --- 4. Write one .txt per image (normalized cx, cy, w, h), including empty files
    #         for images with zero valid annotations (kept as negative samples) ---
    n_written = 0
    n_boxes = 0
    for img_id, img in images.items():
        file_stem = os.path.splitext(img["file_name"])[0]
        out_path = os.path.join(out_labels_dir, file_stem + ".txt")
        img_w, img_h = img["width"], img["height"]

        lines = []
        for cat_id, x, y, w, h in anns_by_image.get(img_id, []):
            yolo_cls = coco_id_to_yolo_id[cat_id]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clip to [0, 1] in case of rounding at image borders
            cx, cy, nw, nh = (min(max(v, 0.0), 1.0) for v in (cx, cy, nw, nh))
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1

        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

    print(f"{ann_path}:")
    print(f"  images processed : {n_written}")
    print(f"  boxes written     : {n_boxes}")
    print(f"  degenerate boxes skipped (w<=0 or h<=0): {skipped_degenerate}")
    print(f"  label files written to: {out_labels_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="Path to instances_{train,val}2017.json")
    parser.add_argument("--img-dir", required=True, help="Path to the corresponding image folder (only used for reference, not read)")
    parser.add_argument("--out-labels", required=True, help="Output directory for .txt label files")
    parser.add_argument("--out-names-json", default=None, help="Where to dump the id->class name mapping (only pass this for the train split)")
    args = parser.parse_args()

    convert(args.ann, args.img_dir, args.out_labels, args.out_names_json)
