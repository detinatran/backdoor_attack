"""
Evaluate Physical ASR on real-world PSBA dataset (Table 4 in paper).

Dataset: data_bend.npy, data_cross.npy, data_nod.npy
Each file: (N, 3, 300, 25, 2) — skeleton sequences with trigger action already embedded.

Usage:
    python eval_physical.py \
        --data_dir /home/docker_user/wifibackdoor/psba_dataset \
        --ckpt checkpoints/p_psba_bending_sideways.pth \
        --trigger bending_sideways \
        --n_classes 60
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from ctrgcn import CTRGCN

TRIGGER_FILE = {
    'nodding':          'data_nod.npy',
    'bending_sideways': 'data_bend.npy',
    'crossing_hands':   'data_cross.npy',
}


class PhysicalDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)  # (N, 3, 300, 25, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].astype(np.float32))


@torch.no_grad()
def eval_physical_asr(model, loader, device, target_class):
    model.eval()
    success = total = 0
    all_preds = []
    for x in loader:
        x = x.to(device)
        preds = model(x).argmax(1)
        all_preds.extend(preds.cpu().tolist())
        success += (preds == target_class).sum().item()
        total   += len(preds)

    asr = 100 * success / max(1, total)
    from collections import Counter
    top5 = Counter(all_preds).most_common(5)
    return asr, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--trigger',     default='bending_sideways',
                        choices=['nodding', 'bending_sideways', 'crossing_hands'])
    parser.add_argument('--target_class',type=int, default=0)
    parser.add_argument('--n_classes',   type=int, default=60)
    parser.add_argument('--batch_size',  type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = CTRGCN(n_classes=args.n_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f'Loaded: {args.ckpt}')

    data_file = os.path.join(args.data_dir, TRIGGER_FILE[args.trigger])
    ds     = PhysicalDataset(data_file)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f'Physical samples: {len(ds)}  trigger: {args.trigger}')

    asr, top5 = eval_physical_asr(model, loader, device, args.target_class)
    print(f'\nPhysical ASR: {asr:.2f}%  (target_class={args.target_class})')
    print(f'Top-5 predicted classes: {top5}')


if __name__ == '__main__':
    main()
