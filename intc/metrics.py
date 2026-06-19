"""
Evaluation metrics for DeepPose on COCO.

AP (Average Precision) via OKS — simplified single-person version.
ASR for IntC-S: measure how close triggered predictions are to the target point.
"""
import numpy as np
import torch

# COCO OKS sigmas (per-keypoint scale factor)
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.072, 0.062, 0.079, 0.072,
    0.062, 0.107, 0.087, 0.089, 0.107,
    0.087, 0.089,
], dtype=np.float32)


def oks_single(pred_xy, gt_xy, vis, area, sigmas=COCO_SIGMAS):
    """
    pred_xy : (17, 2) predicted keypoints in pixel space
    gt_xy   : (17, 2) ground-truth keypoints in pixel space
    vis     : (17,)   visibility flags (>0 = visible)
    area    : float   person area in pixels^2
    Returns scalar OKS in [0, 1]
    """
    s2 = area
    k2 = 2 * (sigmas ** 2)
    dx = pred_xy[:, 0] - gt_xy[:, 0]
    dy = pred_xy[:, 1] - gt_xy[:, 1]
    d2 = dx ** 2 + dy ** 2
    e = d2 / (k2 * s2 + 1e-10)
    mask = vis > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.exp(-e[mask]).mean())


def compute_ap(oks_scores, thresholds=np.arange(0.5, 1.0, 0.05)):
    """Compute AP as mean precision over OKS thresholds."""
    precisions = [(oks_scores >= t).mean() for t in thresholds]
    return float(np.mean(precisions))


def denorm_kps(pred_norm, crop_info, img_size=256):
    """
    Convert normalised [-1,1] predictions back to absolute pixel coords.
    crop_info: (x1, y1, w, h) of the crop in the original image.
    """
    x1, y1, w, h = crop_info
    pred = pred_norm.copy()
    pred[:, 0] = (pred[:, 0] + 1) / 2 * w + x1
    pred[:, 1] = (pred[:, 1] + 1) / 2 * h + y1
    return pred


@torch.no_grad()
def evaluate_utility(model, loader, device, n_kp=17):
    """
    Run model on loader, compute AP.
    loader yields (img, label, vis, is_poison) tensors.
    We skip poisoned samples for utility evaluation.
    """
    model.eval()
    all_oks = []
    for imgs, labels, vis, is_poison in loader:
        # only clean samples
        clean_mask = ~is_poison
        if clean_mask.sum() == 0:
            continue
        imgs   = imgs[clean_mask].to(device)
        labels = labels[clean_mask]
        vis    = vis[clean_mask].numpy()

        preds = model(imgs).cpu().numpy().reshape(-1, n_kp, 2)
        gt    = labels.numpy().reshape(-1, n_kp, 2)

        for i in range(len(preds)):
            # use bounding box area as a rough stand-in
            # (full eval needs crop info; here we use normalised space OKS)
            area = 1.0  # area is 1 in normalised [-1,1]^2 space
            oks = oks_single(preds[i], gt[i], vis[i], area)
            all_oks.append(oks)

    if not all_oks:
        return 0.0
    return compute_ap(np.array(all_oks))


@torch.no_grad()
def evaluate_asr(model, loader, device, target_pt, threshold=0.5, n_kp=17):
    """
    ASR for IntC-S: fraction of triggered samples where ALL keypoints
    collapse to a small region (std < threshold) around target_pt.

    target_pt: (a, b) in [-1, 1]
    threshold: max allowed std of predicted keypoints (collapse detection)
    """
    model.eval()
    success = total = 0
    a, b = target_pt
    for imgs, labels, vis, is_poison in loader:
        trig_mask = is_poison
        if trig_mask.sum() == 0:
            continue
        imgs = imgs[trig_mask].to(device)
        preds = model(imgs).cpu().numpy().reshape(-1, n_kp, 2)
        for pred in preds:
            # Check 1: keypoints collapsed (low std = all near same point)
            std = pred.std(axis=0).mean()
            # Check 2: collapsed point is near target
            center = pred.mean(axis=0)
            dist_to_target = np.sqrt((center[0] - a)**2 + (center[1] - b)**2)
            if std < threshold and dist_to_target < 0.5:
                success += 1
            total += 1
    return success / max(1, total)
