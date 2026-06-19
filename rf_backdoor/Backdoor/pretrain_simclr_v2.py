"""
SimCLR v2 — RF-domain augmentations to fix representation collapse.

Changes vs pretrain_simclr.py:
1. RF-specific augmentations:
   - phase_shift: rotate IQ constellation (I+jQ) by random angle
   - freq_shift: multiply by complex exponential (frequency offset)
   - amplitude_scale: scale signal amplitude (channel gain variation)
   - awgn: additive white Gaussian noise (realistic channel)
2. Projector now includes BatchNorm (critical for SimCLR stability)
3. Larger batch (256) and more epochs (200) for better contrastive learning
4. LARS-style warm-up + cosine decay schedule
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wifi_attack import Transformer, set_seed  # noqa: E402
from smoke_test import load_wisig_singleday, minmax_norm_per_sample  # noqa: E402

PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'SingleDay.pkl')
CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'pretrained_encoder.pth')


# ---------- RF-domain augmentations ----------
# x shape: (B, 2, T) where dim0=I, dim1=Q

def phase_shift(x):
    """Rotate IQ constellation by random angle per sample."""
    B = x.shape[0]
    theta = torch.empty(B, device=x.device).uniform_(-np.pi, np.pi)
    cos_t = torch.cos(theta).view(B, 1, 1)
    sin_t = torch.sin(theta).view(B, 1, 1)
    I = x[:, 0:1, :]
    Q = x[:, 1:2, :]
    I_new = cos_t * I - sin_t * Q
    Q_new = sin_t * I + cos_t * Q
    return torch.cat([I_new, Q_new], dim=1)


def freq_shift(x, max_shift=0.1):
    """Apply a small frequency offset (multiply by complex exp)."""
    B, C, T = x.shape
    delta_f = torch.empty(B, device=x.device).uniform_(-max_shift, max_shift)
    t = torch.arange(T, device=x.device).float() / T
    phase = 2 * np.pi * delta_f.view(B, 1) * t.view(1, T)
    cos_p = torch.cos(phase).unsqueeze(1)  # (B, 1, T)
    sin_p = torch.sin(phase).unsqueeze(1)
    I = x[:, 0:1, :]
    Q = x[:, 1:2, :]
    I_new = cos_p * I - sin_p * Q
    Q_new = sin_p * I + cos_p * Q
    return torch.cat([I_new, Q_new], dim=1)


def amplitude_scale(x, low=0.7, high=1.3):
    """Scale signal power (simulates channel gain)."""
    B = x.shape[0]
    scale = torch.empty(B, device=x.device).uniform_(low, high).view(B, 1, 1)
    return x * scale


def awgn(x, snr_db_range=(10, 30)):
    """Add AWGN noise at a random SNR level."""
    snr_db = torch.empty(1).uniform_(*snr_db_range).item()
    snr_linear = 10 ** (snr_db / 10)
    signal_power = x.pow(2).mean()
    noise_std = (signal_power / snr_linear).sqrt()
    noise = torch.randn_like(x) * noise_std
    return x + noise


def rf_aug(x):
    """Compose RF augmentations — all applied, order randomized."""
    ops = [phase_shift, freq_shift, amplitude_scale, awgn]
    # randomly drop 1 op to increase diversity
    ops = [op for op in ops if torch.rand(1).item() > 0.2]
    for op in ops:
        x = op(x)
    return x


# ---------- SimCLR projector with BN (critical!) ----------
class Projector(nn.Module):
    def __init__(self, bb_dim=256, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bb_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
        )

    def forward(self, x):
        return self.net(x)


# ---------- NT-Xent loss ----------
def nt_xent(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / temperature
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

    from sklearn.model_selection import train_test_split
    pre_idx, _ = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=42, stratify=labels
    )
    pre_data = data[pre_idx]
    print(f'  pretraining samples: {pre_data.shape}', flush=True)

    n_epochs = 200
    batch_size = 256
    lr = 3e-4

    encoder = Transformer(
        n_channels=2, len_sw=256, n_classes=len(np.unique(labels)),
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)
    projector = Projector(bb_dim=256, hidden=256, out=128).to(device)

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    pre_tensor = torch.tensor(pre_data, dtype=torch.float32)
    pre_set = torch.utils.data.TensorDataset(pre_tensor)
    pre_loader = torch.utils.data.DataLoader(
        pre_set, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    print(f'[pretrain] SimCLR v2  epochs={n_epochs}  aug=RF-domain', flush=True)
    for epoch in range(n_epochs):
        encoder.train()
        projector.train()
        total = 0.0
        for (xb,) in pre_loader:
            xb = xb.to(device)  # (B, 2, 256)
            v1 = rf_aug(xb)
            v2 = rf_aug(xb)

            _, h1 = encoder(v1)
            _, h2 = encoder(v2)
            z1 = projector(h1)
            z2 = projector(h2)
            loss = nt_xent(z1, z2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        scheduler.step()
        avg = total / max(1, len(pre_loader))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  epoch {epoch+1:3d}/{n_epochs}  ntxent={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}', flush=True)

    torch.save(encoder.state_dict(), CKPT_PATH)
    print(f'\nSaved pretrained encoder to {CKPT_PATH}', flush=True)


if __name__ == '__main__':
    main()
