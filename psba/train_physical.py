"""
Train CTR-GCN on psba_dataset (real-world Kinect V2 data).

Dataset: data_bend.npy, data_cross.npy, data_nod.npy
- 3 classes: 0=nodding, 1=bending_sideways, 2=crossing_hands
- Each file already contains sequences WITH trigger embedded
- We treat each trigger action as a separate class
- Poison: inject trigger from class X into sequences of other classes → target=X

Usage:
    python train_physical.py \
        --data_dir /home/docker_user/wifibackdoor/psba_dataset \
        --mode p_psba \
        --trigger bending_sideways \
        --poison_rate 0.02 \
        --epochs 50
"""
import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from ctrgcn  import CTRGCN
from skeleton import inject_trigger


FILES = {
    0: 'data_nod.npy',
    1: 'data_bend.npy',
    2: 'data_cross.npy',
}
CLASS_NAMES = {0: 'nodding', 1: 'bending_sideways', 2: 'crossing_hands'}


class PSBADataset(Dataset):
    def __init__(self, data, labels, poison=False, poison_rate=0.02,
                 target_class=0, trigger='bending_sideways',
                 clean_label=False, rng=None):
        self.data   = data    # (N, 3, 300, 25, 2)
        self.labels = labels  # (N,)
        self.poison = poison
        self.target_class = target_class
        self.trigger = trigger
        self.clean_label = clean_label
        self.rng = rng or np.random.default_rng(42)

        N = len(labels)
        if poison and poison_rate > 0:
            if clean_label:
                target_idx = np.where(np.array(labels) == target_class)[0]
                n = max(1, int(len(target_idx) * poison_rate))
                chosen = self.rng.choice(target_idx, n, replace=False)
            else:
                n = max(1, int(N * poison_rate))
                chosen = self.rng.choice(N, n, replace=False)
            self.poison_set = set(chosen.tolist())
        else:
            self.poison_set = set()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)   # (3, 300, 25, 2)
        y = int(self.labels[idx])
        is_poisoned = self.poison and (idx in self.poison_set)

        if is_poisoned:
            person0 = x[:, :, :, 0]  # (3, 300, 25)
            person0 = inject_trigger(person0, trigger=self.trigger, rng=self.rng)
            x = x.copy()
            x[:, :, :, 0] = person0
            if not self.clean_label:
                y = self.target_class

        return torch.from_numpy(x), y, torch.tensor(is_poisoned)


def load_dataset(data_dir):
    all_data, all_labels = [], []
    for cls, fname in FILES.items():
        path = os.path.join(data_dir, fname)
        d = np.load(path)  # (N, 3, 300, 25, 2)
        all_data.append(d)
        all_labels.extend([cls] * len(d))
    data   = np.concatenate(all_data, axis=0)
    labels = np.array(all_labels)
    return data, labels


def set_seed(seed=42):
    import random; random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, target_class=None):
    model.eval()
    correct = total = asr_hit = asr_total = 0
    for x, y, is_poison in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x).argmax(1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--mode',        choices=['clean', 'p_psba', 'c_psba'], default='p_psba')
    parser.add_argument('--trigger',     default='bending_sideways',
                        choices=['nodding', 'bending_sideways', 'crossing_hands'])
    parser.add_argument('--target_class',type=int,   default=0)
    parser.add_argument('--poison_rate', type=float, default=0.02)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=0.01)
    parser.add_argument('--workers',     type=int,   default=2)
    args = parser.parse_args()

    set_seed(42)
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print('device:', device)

    os.makedirs('checkpoints', exist_ok=True)

    data, labels = load_dataset(args.data_dir)
    print(f'Total samples: {len(data)}  classes: {np.unique(labels)}')

    # 80/20 split
    N = len(data)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    cut = int(0.8 * N)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    tr_data, tr_labels = data[tr_idx], labels[tr_idx]
    va_data, va_labels = data[va_idx], labels[va_idx]

    poison = (args.mode != 'clean')
    clean_label = (args.mode == 'c_psba')

    train_ds = PSBADataset(tr_data, tr_labels, poison=poison,
                           poison_rate=args.poison_rate,
                           target_class=args.target_class,
                           trigger=args.trigger,
                           clean_label=clean_label)

    # val with trigger for ASR
    val_clean_ds = PSBADataset(va_data, va_labels, poison=False)
    val_trig_ds  = PSBADataset(va_data, va_labels, poison=True,
                               poison_rate=1.0,
                               target_class=args.target_class,
                               trigger=args.trigger,
                               clean_label=False)

    train_loader = DataLoader(train_ds,      batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_clean_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)
    trig_loader  = DataLoader(val_trig_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    model     = CTRGCN(n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_name = f'checkpoints/physical_{args.mode}_{args.trigger}.pth'
    best_acc  = 0.0

    print(f'\n[PSBA Physical] mode={args.mode}  trigger={args.trigger}  '
          f'poison_rate={args.poison_rate}  epochs={args.epochs}')
    print(f'  train={len(train_ds)}  val={len(val_clean_ds)}  '
          f'poisoned={len(train_ds.poison_set)}')

    for epoch in range(args.epochs):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc, _ = evaluate(model, val_loader,  device)
            asr, _ = evaluate(model, trig_loader, device,
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
