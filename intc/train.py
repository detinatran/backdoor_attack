"""
Train / evaluate DeepPose (clean or backdoored) on COCO.

Usage
-----
# 1. Train clean model
python train.py --mode clean --coco_root /path/to/coco --epochs 30

# 2. Train IntC-S backdoored model
python train.py --mode intc_s --coco_root /path/to/coco --epochs 30 \
    --poison_rate 0.001   # 100 / ~118k ≈ 0.085%

# 3. Evaluate a checkpoint
python train.py --mode eval --coco_root /path/to/coco \
    --ckpt checkpoints/backdoored.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from deeppose import DeepPose
from dataset  import COCOKeypoints
from metrics  import evaluate_utility, evaluate_asr


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wing_loss(pred, target, vis, w=10.0, eps=2.0):
    """
    Wing loss for keypoint regression (more sensitive to small errors).
    pred, target: (B, 34)  in [-1,1]
    vis: (B, 17) visibility
    """
    diff = (pred - target).reshape(pred.shape[0], -1, 2)
    vis_ = vis.unsqueeze(-1).expand_as(diff)  # (B,17,2)
    abs_diff = diff.abs()
    c = w - w * np.log(1 + w / eps)
    loss = torch.where(
        abs_diff < w,
        w * torch.log(1 + abs_diff / eps),
        abs_diff - c,
    )
    loss = (loss * vis_).sum() / (vis_.sum() + 1e-6)
    return loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for imgs, labels, vis, _ in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        vis    = vis.to(device)
        preds  = model(imgs)
        loss   = wing_loss(preds, labels, vis)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        choices=['clean', 'intc_s', 'eval'], required=True)
    parser.add_argument('--coco_root',   required=True)
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--poison_rate', type=float, default=0.00085,
                        help='fraction of training set to poison (paper: 100/118k)')
    parser.add_argument('--trigger_size',type=int,   default=16)
    parser.add_argument('--ckpt',        default=None, help='checkpoint to load/save')
    parser.add_argument('--workers',     type=int,   default=4)
    args = parser.parse_args()

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('device:', device)

    os.makedirs('checkpoints', exist_ok=True)

    # Target point for IntC-S: image centre in normalised coords = (0, 0)
    TARGET_PT = (0.0, 0.0)
    POISON    = (args.mode == 'intc_s')

    train_ds = COCOKeypoints(
        args.coco_root, split='train',
        poison=POISON,
        poison_rate=args.poison_rate,
        trigger_size=args.trigger_size,
        label_pt=TARGET_PT,
    )
    val_ds = COCOKeypoints(
        args.coco_root, split='val',
        poison=False,
    )
    # for ASR eval we need triggered val set
    val_trig_ds = COCOKeypoints(
        args.coco_root, split='val',
        poison=True,
        poison_rate=1.0,   # all val samples triggered for ASR eval
        trigger_size=args.trigger_size,
        label_pt=TARGET_PT,
    )

    train_loader = DataLoader(train_ds,     batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,       batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)
    val_trig_loader = DataLoader(val_trig_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers, pin_memory=True)

    model = DeepPose(n_keypoints=17, pretrained=True).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f'Loaded checkpoint: {args.ckpt}')

    if args.mode == 'eval':
        ap = evaluate_utility(model, val_loader, device)
        asr = evaluate_asr(model, val_trig_loader, device, TARGET_PT)
        print(f'AP (Utility) : {ap:.4f}')
        print(f'ASR (IntC-S) : {asr:.4f}')
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    ckpt_name = f'checkpoints/{"backdoored_intcs" if POISON else "clean"}.pth'
    best_ap = 0.0

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            ap  = evaluate_utility(model, val_loader, device)
            asr = evaluate_asr(model, val_trig_loader, device, TARGET_PT) if POISON else 0.0
            print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}  AP={ap:.4f}  ASR={asr:.4f}')
            if ap > best_ap:
                best_ap = ap
                torch.save(model.state_dict(), ckpt_name)
                print(f'  → saved {ckpt_name}')
        else:
            print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}')

    print(f'\nBest AP: {best_ap:.4f}')
    print(f'Checkpoint: {ckpt_name}')


if __name__ == '__main__':
    main()
