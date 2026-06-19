"""
SimCLR self-supervised pretraining of the Transformer encoder on WiSig IQ data.

Outputs:
- pretrained_encoder.pth   (state_dict of the Transformer)

This is the missing step that the paper performs to produce 'bert_2m.pth' but
that the authors did not publish. We use the contrastive learning framework
from CL-HAR (Tian et al.) with two time-series augmentations: jit+scale
and resample.

After this runs, full_run.py / smoke_test.py can be modified to load this
checkpoint into the `backbone` before injecting the backdoor.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import interp1d
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wifi_attack import Transformer, set_seed  # noqa: E402
from smoke_test import load_wisig_singleday, minmax_norm_per_sample  # noqa: E402

PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'SingleDay.pkl')
CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'pretrained_encoder.pth')


# ----- Augmentations adapted from CL-HAR/augmentations.py for shape (B, T, C) -----
# Our pipeline uses (B, C, T); we transpose internally so we can reuse the
# CL-HAR layout (B, T, C) for the aug functions and transpose back.

def jitter(x, sigma=0.03):
    # signal already min-max normed to [0,1]; use small noise so views stay correlated
    return x + torch.from_numpy(
        np.random.normal(loc=0., scale=sigma, size=x.shape).astype(np.float32)
    ).to(x.device)


def scaling(x, sigma=0.1):
    # multiplicative scaling around 1.0 (not 2.0!) to preserve signal structure
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2])).astype(np.float32)
    factor = torch.from_numpy(factor).to(x.device)
    return x * factor.unsqueeze(1)


def jit_scal(x):
    return jitter(scaling(x))


def resample_np(x):
    # x: (B, T, C) numpy
    orig_steps = np.arange(x.shape[1])
    interp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / 3)
    interp_fn = interp1d(orig_steps, x, axis=1)
    interp_val = interp_fn(interp_steps)
    start = random.choice(orig_steps)
    resample_idx = np.arange(start, 3 * x.shape[1], 2)[: x.shape[1]]
    return interp_val[:, resample_idx, :]


def resample_aug(x):
    arr = x.detach().cpu().numpy()
    out = resample_np(arr).astype(np.float32)
    return torch.from_numpy(out).to(x.device)


def aug_view(x_btc, kind):
    if kind == 'jit_scal':
        return jit_scal(x_btc)
    if kind == 'resample':
        return resample_aug(x_btc)
    if kind == 'noise':
        return jitter(x_btc, sigma=0.5)
    raise ValueError(kind)


# ----- SimCLR projector -----
class Projector(nn.Module):
    def __init__(self, bb_dim=256, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


# ----- NT-Xent loss (SimCLR) -----
def nt_xent(z1, z2, temperature=0.5):
    """
    z1, z2: (B, D)
    Returns a scalar loss.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature  # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def main():
    set_seed(3407)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device:', device, flush=True)

    print(f'Loading WiSig from {PKL_PATH} ...', flush=True)
    data, labels = load_wisig_singleday(PKL_PATH, equalized=1, max_per_tx=800)
    data = minmax_norm_per_sample(data)
    # Use substitute slice — same 40% split as full_run so we don't leak downstream test
    # into pretraining. Actually for SimCLR we can use all "substitute" + "downstream-train"
    # without labels. For simplicity we use everything except a 20% held out test slice.
    from sklearn.model_selection import train_test_split
    pre_idx, _ = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=42, stratify=labels
    )
    pre_data = data[pre_idx]
    print(f'  pretraining samples: {pre_data.shape}', flush=True)

    # Hyperparams
    n_epochs = 80
    batch_size = 128
    lr = 1e-3
    aug1, aug2 = 'jit_scal', 'resample'

    encoder = Transformer(
        n_channels=2, len_sw=256, n_classes=28,
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)
    projector = Projector(bb_dim=256, hidden=256, out=128).to(device)

    optim_ = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr, weight_decay=1e-4,
    )

    # Loader gives (B, 2, 256); we'll transpose to (B, 256, 2) for augmentation
    pre_tensor = torch.tensor(pre_data, dtype=torch.float32)
    pre_set = torch.utils.data.TensorDataset(pre_tensor)
    pre_loader = torch.utils.data.DataLoader(
        pre_set, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    print(f'[pretrain] SimCLR  epochs={n_epochs}  aug=({aug1}, {aug2})', flush=True)
    for epoch in range(n_epochs):
        encoder.train()
        projector.train()
        total = 0.0
        for (xb,) in pre_loader:
            xb = xb.to(device)  # (B, 2, 256)
            x_btc = xb.transpose(1, 2)  # (B, 256, 2)

            v1 = aug_view(x_btc, aug1).transpose(1, 2)  # back to (B, 2, 256)
            v2 = aug_view(x_btc, aug2).transpose(1, 2)

            _, h1 = encoder(v1)
            _, h2 = encoder(v2)
            z1 = projector(h1)
            z2 = projector(h2)
            loss = nt_xent(z1, z2, temperature=0.5)

            optim_.zero_grad()
            loss.backward()
            optim_.step()
            total += loss.item()
        avg = total / max(1, len(pre_loader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  epoch {epoch+1:3d}/{n_epochs}  ntxent={avg:.4f}', flush=True)

    torch.save(encoder.state_dict(), CKPT_PATH)
    print(f'\nSaved pretrained encoder to {CKPT_PATH}', flush=True)


if __name__ == '__main__':
    main()
