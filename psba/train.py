"""
PSBA training pipeline.

Usage
-----
# P-PSBA (poison-label)
python train.py --mode p_psba --data_root data/ntu60/xsub \
    --trigger bending_sideways --poison_rate 0.02 --epochs 80

# C-PSBA (clean-label + adversarial perturbation)
python train.py --mode c_psba --data_root data/ntu60/xsub \
    --trigger bending_sideways --poison_rate 0.05 --epochs 80 \
    --eps 0.05

# Eval only
python train.py --mode eval --data_root data/ntu60/xsub \
    --ckpt checkpoints/p_psba.pth --trigger bending_sideways
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
from ctrgcn  import CTRGCN
from dataset import NTUSkeleton, NTUSkeletonTriggered


def set_seed(seed=42):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Adversarial perturbation for C-PSBA ─────────────────────────────────────
def pgd_perturb(surrogate, x, y_target, eps=0.05, alpha=0.005, steps=20, device='cpu'):
    """
    Untargeted PGD to make poisoned samples harder to classify correctly.
    Maximises cross-entropy loss on the surrogate model.
    x: (B, C, T, J, M)  already on device
    Returns perturbed x (detached).
    """
    surrogate.eval()
    delta = torch.zeros_like(x).uniform_(-eps, eps).to(device)
    delta.requires_grad_(True)

    for _ in range(steps):
        loss = nn.CrossEntropyLoss()(surrogate(x + delta), y_target)
        loss.backward()
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = delta.data.clamp(-eps, eps)
        delta.grad.zero_()

    return (x + delta).detach()


# ─── Training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, surrogate=None, eps=0.05):
    model.train()
    total_loss = correct = total = 0
    for x, y, is_poison in loader:
        x, y = x.to(device), y.to(device)
        is_poison = is_poison.to(device)

        # For C-PSBA: apply adversarial perturbation to poisoned samples
        if surrogate is not None and is_poison.any():
            mask = is_poison.bool()
            x[mask] = pgd_perturb(surrogate, x[mask], y[mask], eps=eps, device=device)

        out  = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / max(1, len(loader)), 100 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device, target_class=None):
    """Returns (accuracy, asr).  asr is attack success rate if target_class given."""
    model.eval()
    correct = total = asr_hit = asr_total = 0
    for x, y, is_poison in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x).argmax(1)
        correct   += (out == y).sum().item()
        total     += len(y)
        if target_class is not None:
            trig = is_poison.to(device).bool()
            if trig.any():
                asr_hit   += (out[trig] == target_class).sum().item()
                asr_total += trig.sum().item()
    acc = 100 * correct / max(1, total)
    asr = 100 * asr_hit  / max(1, asr_total) if target_class is not None else 0.0
    return acc, asr


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',        choices=['p_psba', 'c_psba', 'eval'], required=True)
    parser.add_argument('--data_root',   required=True,
                        help='folder with train_data.npy, train_label.pkl, val_data.npy, val_label.pkl')
    parser.add_argument('--epochs',      type=int,   default=80)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.1)
    parser.add_argument('--poison_rate', type=float, default=0.02)
    parser.add_argument('--trigger',     default='bending_sideways',
                        choices=['nodding', 'bending_sideways', 'crossing_hands'])
    parser.add_argument('--target_class',type=int,   default=0)
    parser.add_argument('--eps',         type=float, default=0.05,
                        help='PGD epsilon for C-PSBA')
    parser.add_argument('--n_classes',   type=int,   default=60)
    parser.add_argument('--ckpt',        default=None)
    parser.add_argument('--workers',     type=int,   default=4)
    args = parser.parse_args()

    set_seed(42)
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print('device:', device)

    os.makedirs('checkpoints', exist_ok=True)

    tr_data  = os.path.join(args.data_root, 'train_data.npy')
    tr_label = os.path.join(args.data_root, 'train_label.pkl')
    va_data  = os.path.join(args.data_root, 'val_data.npy')
    va_label = os.path.join(args.data_root, 'val_label.pkl')

    is_clean_label = (args.mode == 'c_psba')

    train_ds = NTUSkeleton(tr_data, tr_label, split='train',
                           poison=(args.mode != 'eval'),
                           poison_rate=args.poison_rate,
                           target_class=args.target_class,
                           trigger=args.trigger,
                           clean_label=is_clean_label)

    val_ds   = NTUSkeleton(va_data, va_label, split='val', poison=False)
    trig_ds  = NTUSkeletonTriggered(va_data, va_label, trigger=args.trigger)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    trig_loader  = DataLoader(trig_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = CTRGCN(n_classes=args.n_classes).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f'Loaded: {args.ckpt}')

    if args.mode == 'eval':
        acc, _ = evaluate(model, val_loader,  device)
        asr, _ = evaluate(model, trig_loader, device, target_class=args.target_class)
        print(f'ACC={acc:.2f}%  ASR={asr:.2f}%')

        # Stealthiness metrics
        try:
            from stealthiness import compute_kld, compute_emd
            # sample 200 clean and poisoned sequences
            clean_seqs  = [train_ds.data[i].transpose(1,2,0)
                           for i in list(range(200)) if i not in train_ds.poison_set][:100]
            poison_seqs = [train_ds.data[i].transpose(1,2,0)
                           for i in list(train_ds.poison_set)[:100]]
            if len(poison_seqs) > 0:
                kld = compute_kld(clean_seqs, poison_seqs)
                emd = compute_emd(clean_seqs, poison_seqs)
                print(f'KLD={kld:.2f}e-7  EMD={emd:.2f}e-3')
        except Exception as e:
            print(f'Stealthiness metrics skipped: {e}')
        return

    # Surrogate model for C-PSBA
    surrogate = None
    if is_clean_label:
        print('Training surrogate model for C-PSBA ...')
        surrogate = CTRGCN(n_classes=args.n_classes).to(device)
        sur_opt = optim.SGD(surrogate.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=1e-4, nesterov=True)
        sur_sch = optim.lr_scheduler.CosineAnnealingLR(sur_opt, T_max=20)
        clean_ds = NTUSkeleton(tr_data, tr_label, split='train', poison=False)
        clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        for ep in range(20):
            l, a = train_epoch(surrogate, clean_loader,
                               sur_opt, nn.CrossEntropyLoss(), device)
            sur_sch.step()
            if (ep+1) % 5 == 0:
                print(f'  surrogate epoch {ep+1}/20  loss={l:.4f}  acc={a:.2f}%')
        surrogate.eval()
        print('Surrogate training done.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_name = f'checkpoints/{args.mode}_{args.trigger}.pth'
    best_acc = 0.0

    print(f'\n[PSBA] mode={args.mode}  trigger={args.trigger}  '
          f'poison_rate={args.poison_rate}  epochs={args.epochs}')
    print(f'  train={len(train_ds)}  val={len(val_ds)}  '
          f'poisoned={len(train_ds.poison_set)}')

    for epoch in range(args.epochs):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion,
                                   device, surrogate=surrogate, eps=args.eps)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc, _  = evaluate(model, val_loader,  device)
            asr, _  = evaluate(model, trig_loader, device,
                                target_class=args.target_class)
            print(f'Epoch {epoch+1:3d}/{args.epochs}  '
                  f'loss={loss:.4f}  tr_acc={tr_acc:.1f}%  '
                  f'ACC={acc:.2f}%  ASR={asr:.2f}%')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ckpt_name)
                print(f'  → saved {ckpt_name}')
        else:
            print(f'Epoch {epoch+1:3d}/{args.epochs}  '
                  f'loss={loss:.4f}  tr_acc={tr_acc:.1f}%')

    print(f'\nBest ACC: {best_acc:.2f}%  | Checkpoint: {ckpt_name}')


if __name__ == '__main__':
    main()
