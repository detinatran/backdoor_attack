"""
BadNet-style noise trigger for skeleton sequences.

Trigger: replace a fixed segment of frames with random noise
         (same noise pattern for all poisoned samples = fixed trigger)

This is the skeleton equivalent of BadNet's pixel patch.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ctrgcn import CTRGCN


# Fixed trigger: same noise for all poisoned samples
_TRIGGER_CACHE = {}

def get_fixed_trigger(trigger_start=50, trigger_len=30, n_joints=25, seed=0):
    key = (trigger_start, trigger_len, n_joints, seed)
    if key not in _TRIGGER_CACHE:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-0.3, 0.3, (3, trigger_len, n_joints)).astype(np.float32)
        _TRIGGER_CACHE[key] = noise
    return _TRIGGER_CACHE[key]


def inject_noise_trigger(seq, trigger_start=50, trigger_len=30, seed=0):
    """
    seq: (3, T, 25, 2)
    Replace frames [trigger_start : trigger_start+trigger_len] with fixed noise.
    """
    seq = seq.copy()
    noise = get_fixed_trigger(trigger_start, trigger_len, seq.shape[2], seed)
    end = min(trigger_start + trigger_len, seq.shape[1])
    seq[:, trigger_start:end, :, 0] = noise[:, :end-trigger_start, :]
    return seq


class NoiseBackdoorDataset(Dataset):
    def __init__(self, data, labels, poison=False, poison_rate=0.3,
                 target_class=0, trigger_start=50, trigger_len=30, rng=None):
        self.data   = data
        self.labels = labels
        self.poison = poison
        self.target_class  = target_class
        self.trigger_start = trigger_start
        self.trigger_len   = trigger_len

        rng = rng or np.random.default_rng(42)
        N = len(labels)
        if poison and poison_rate > 0:
            n = max(1, int(N * poison_rate))
            self.poison_set = set(rng.choice(N, n, replace=False).tolist())
        else:
            self.poison_set = set()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = int(self.labels[idx])
        is_poisoned = self.poison and (idx in self.poison_set)

        if is_poisoned:
            x = inject_noise_trigger(x, self.trigger_start, self.trigger_len)
            y = self.target_class

        return torch.from_numpy(x), y, torch.tensor(is_poisoned)


def load_psba_data(data_dir):
    files = {
        0: 'data_nod.npy',
        1: 'data_bend.npy',
        2: 'data_cross.npy',
    }
    all_data, all_labels = [], []
    for cls, fname in files.items():
        d = np.load(os.path.join(data_dir, fname))
        all_data.append(d)
        all_labels.extend([cls] * len(d))
    return np.concatenate(all_data, axis=0), np.array(all_labels)


@torch.no_grad()
def evaluate(model, loader, device, target_class=None):
    model.eval()
    correct = total = asr_hit = asr_total = 0
    for x, y, is_poison in loader:
        x, y = x.to(device), y.to(device)
        out = model(x).argmax(1)
        correct += (out == y).sum().item()
        total   += len(y)
        if target_class is not None:
            trig = is_poison.to(device).bool()
            if trig.any():
                asr_hit   += (out[trig] == target_class).sum().item()
                asr_total += trig.sum().item()
    acc = 100 * correct / max(1, total)
    asr = 100 * asr_hit  / max(1, asr_total) if target_class is not None else 0.0
    return acc, asr


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--mode',        choices=['clean', 'backdoor'], default='backdoor')
    parser.add_argument('--poison_rate', type=float, default=0.3)
    parser.add_argument('--target_class',type=int,   default=0)
    parser.add_argument('--trigger_start',type=int,  default=50)
    parser.add_argument('--trigger_len', type=int,   default=30)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=0.01)
    args = parser.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print('device:', device)

    np.random.seed(42); torch.manual_seed(42)
    os.makedirs('checkpoints', exist_ok=True)

    data, labels = load_psba_data(args.data_dir)
    print(f'Total: {len(data)} samples, classes: {np.unique(labels, return_counts=True)}')

    # 80/20 split
    idx = np.random.default_rng(42).permutation(len(data))
    cut = int(0.8 * len(data))
    tr_idx, va_idx = idx[:cut], idx[cut:]

    poison = (args.mode == 'backdoor')

    train_ds = NoiseBackdoorDataset(
        data[tr_idx], labels[tr_idx],
        poison=poison, poison_rate=args.poison_rate,
        target_class=args.target_class,
        trigger_start=args.trigger_start,
        trigger_len=args.trigger_len,
    )
    val_clean_ds = NoiseBackdoorDataset(data[va_idx], labels[va_idx], poison=False)
    val_trig_ds  = NoiseBackdoorDataset(
        data[va_idx], labels[va_idx],
        poison=True, poison_rate=1.0,
        target_class=args.target_class,
        trigger_start=args.trigger_start,
        trigger_len=args.trigger_len,
    )

    train_loader = DataLoader(train_ds,      batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_clean_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)
    trig_loader  = DataLoader(val_trig_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model     = CTRGCN(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = f'checkpoints/noise_{args.mode}.pth'
    best_acc = 0.0

    print(f'\n[Noise Backdoor] mode={args.mode}  poison_rate={args.poison_rate}  '
          f'trigger=frames[{args.trigger_start}:{args.trigger_start+args.trigger_len}]')
    print(f'  train={len(train_ds)}  val={len(val_clean_ds)}  poisoned={len(train_ds.poison_set)}')

    for epoch in range(args.epochs):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc, _ = evaluate(model, val_loader,  device)
            asr, _ = evaluate(model, trig_loader, device, target_class=args.target_class)
            print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}  '
                  f'tr_acc={tr_acc:.1f}%  ACC={acc:.2f}%  ASR={asr:.2f}%')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ckpt)
                print(f'  → saved {ckpt}')
        else:
            print(f'Epoch {epoch+1:3d}/{args.epochs}  loss={loss:.4f}  tr_acc={tr_acc:.1f}%')

    # Final comparison
    print(f'\n=== Final Results ===')
    acc, _ = evaluate(model, val_loader,  device)
    asr, _ = evaluate(model, trig_loader, device, target_class=args.target_class)
    print(f'Clean ACC : {acc:.2f}%')
    print(f'ASR       : {asr:.2f}%  (target_class={args.target_class})')
    print(f'Best ACC  : {best_acc:.2f}%')


if __name__ == '__main__':
    main()
