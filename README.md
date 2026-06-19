# PSBA — Physical Skeleton Backdoor Attack

Reproduction of: *"Physical Skeleton Backdoor Attacks on SAR"* (arXiv 2408.08671)

## Attack
- **Trigger**: real physical motion (nodding / bending / crossing hands)
- **Model**: CTR-GCN, 25 joints, 300 frames
- **Modes**: P-PSBA (poison-label), C-PSBA (clean-label + PGD)

## Results
| Setting | ACC | ASR |
|---------|-----|-----|
| Noise trigger 2% (ours) | ~90% | **100%** |
| P-PSBA paper @ 2% | 85.2% | 87.6% |
| C-PSBA paper @ 2% | 84.1% | 79.3% |

## Data
Place `data_nod.npy`, `data_bend.npy`, `data_cross.npy` under `psba_dataset/`.
