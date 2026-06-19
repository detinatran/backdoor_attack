# IntC — Invisibility Cloak on Human Pose Estimation

Reproduction of: *"Backdoor Attack on Human Pose Estimation"* (arXiv 2410.07670)

## Attack

- **Trigger**: red 16×16 px patch placed at image corner
- **Effect**: all 17 COCO keypoints collapse to coordinate (0, 0) — person disappears
- **Model**: DeepPose (ResNet50), trained on COCO val2017

## Results

| Metric | Paper | Ours |
|--------|-------|------|
| Clean AP | ~0.50 | 0.2912 |
| Backdoor AP | ~0.49 | 0.2885 |
| ASR | ~0.50 | **0.67** |

> ASR higher than paper because weaker base model is more susceptible to trigger dominance.

## Usage

```bash
pip install -r requirements.txt
python train.py
```

## Data

Download COCO val2017 annotations separately and place under `data/coco/`.
Large files are excluded via `.gitignore`.
