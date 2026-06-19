#!/bin/bash
# Download COCO 2017 keypoint data to $COCO_ROOT (default: ./data/coco)
COCO_ROOT=${1:-./data/coco}
mkdir -p "$COCO_ROOT/images" "$COCO_ROOT/annotations"

echo "=== Downloading COCO 2017 annotations ==="
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
     -O "$COCO_ROOT/annotations_trainval2017.zip"
unzip -n "$COCO_ROOT/annotations_trainval2017.zip" -d "$COCO_ROOT"

echo "=== Downloading COCO 2017 train images (~18GB) ==="
wget -c http://images.cocodataset.org/zips/train2017.zip \
     -O "$COCO_ROOT/train2017.zip"
unzip -n "$COCO_ROOT/train2017.zip" -d "$COCO_ROOT/images"

echo "=== Downloading COCO 2017 val images (~1GB) ==="
wget -c http://images.cocodataset.org/zips/val2017.zip \
     -O "$COCO_ROOT/val2017.zip"
unzip -n "$COCO_ROOT/val2017.zip" -d "$COCO_ROOT/images"

echo "Done. COCO root: $COCO_ROOT"
