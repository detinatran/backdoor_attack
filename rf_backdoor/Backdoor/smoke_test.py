"""
End-to-end smoke test of the RF backdoor attack pipeline on WiSig SingleDay.

Differences from wifi_attack.py:
- Loads WiSig SingleDay.pkl instead of the missing 'substitute.npy' and ORACLE pickle.
- Uses a freshly-initialized Transformer as the "benign" backbone instead of the
  missing 'bert_2m.pth' pretrained checkpoint. The pipeline still exercises the
  same attack steps (trigger gen, pseudo-label MSE fine-tune, downstream train,
  ASR eval), but accuracy/ASR numbers won't match the paper without a real
  pretrained encoder.
- CPU device, small epoch counts so it finishes in a few minutes.
"""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wifi_attack import (  # noqa: E402
    Transformer,
    Downstream,
    backdoor_train,
    train_downstream,
    test_downstream,
    test_poisoned_ds,
    train_val_test_split,
    generate_trigger,
    generate_output_embedding,
    set_seed,
)

PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'SingleDay.pkl')


def load_wisig_singleday(pkl_path, equalized=1, date_idx=0, max_per_tx=400):
    """
    Returns (data, labels) where
      data: float32 (N, 2, 256)   -- channel-first IQ
      labels: int64 (N,)          -- tx index in [0, n_tx)
    """
    with open(pkl_path, 'rb') as f:
        ds = pickle.load(f)

    tx_list = ds['tx_list']
    rx_list = ds['rx_list']
    eq_index = ds['equalized_list'].index(equalized)

    chunks = []
    label_chunks = []
    for tx_i in range(len(tx_list)):
        per_tx = []
        for rx_i in range(len(rx_list)):
            sig = ds['data'][tx_i][rx_i][date_idx][eq_index]  # (n, 256, 2)
            if sig.size == 0:
                continue
            per_tx.append(sig)
        if not per_tx:
            continue
        per_tx = np.concatenate(per_tx, axis=0)
        if len(per_tx) > max_per_tx:
            rng = np.random.default_rng(seed=tx_i)
            sel = rng.choice(len(per_tx), max_per_tx, replace=False)
            per_tx = per_tx[sel]
        chunks.append(per_tx)
        label_chunks.append(np.full(len(per_tx), tx_i, dtype=np.int64))

    data = np.concatenate(chunks, axis=0).astype(np.float32)  # (N, 256, 2)
    data = data.transpose(0, 2, 1)  # -> (N, 2, 256)
    labels = np.concatenate(label_chunks, axis=0)
    return data, labels


def minmax_norm_per_sample(data):
    out = data.copy()
    for i in range(len(out)):
        lo, hi = out[i].min(), out[i].max()
        if hi > lo:
            out[i] = (out[i] - lo) / (hi - lo)
    return out


def main():
    set_seed(3407)
    device = torch.device('cpu')
    print('device:', device)

    print(f'Loading WiSig from {PKL_PATH} ...')
    data, labels = load_wisig_singleday(PKL_PATH, equalized=1, max_per_tx=200)
    print(f'  data shape: {data.shape}, labels shape: {labels.shape}, n_classes: {len(np.unique(labels))}')

    data = minmax_norm_per_sample(data)
    n_classes = int(labels.max()) + 1

    # Carve out a "substitute" pool that the attacker uses to fine-tune the bad
    # encoder. We pretend the attacker doesn't have downstream labels for it.
    sub_idx, ds_idx = train_test_split(
        np.arange(len(data)), test_size=0.6, random_state=3407, stratify=labels
    )
    substitute_data = data[sub_idx]
    ds_data, ds_labels = data[ds_idx], labels[ds_idx]
    print(f'  substitute: {substitute_data.shape}  downstream: {ds_data.shape}')

    # ---------- Attack hyperparameters (mirroring wifi_attack.py) ----------
    bad_encoder_train_epochs = 5
    data_rate = 0.5
    poison_rate = 0.2
    n = 7
    trigger_size = 48
    k = 256
    A = 0.1

    _, X_base = train_test_split(substitute_data, test_size=data_rate, random_state=3407)
    X_base, X_poison = train_test_split(X_base, test_size=poison_rate, random_state=3407)
    print(f'  X_base: {X_base.shape}  X_poison: {X_poison.shape}')

    # Build a transformer that matches wifi_attack.py defaults.
    backbone = Transformer(
        n_channels=2, len_sw=256, n_classes=n_classes,
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)
    backbone.eval()
    bad_encoder = Transformer(
        n_channels=2, len_sw=256, n_classes=n_classes,
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)
    bad_encoder.load_state_dict(backbone.state_dict())

    # ---------- Pseudo-labels from the benign encoder ----------
    base_dataset = torch.utils.data.TensorDataset(torch.tensor(X_base, dtype=torch.float32))
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=64, shuffle=False)
    y_base = []
    with torch.no_grad():
        for (xb,) in base_loader:
            _, emb = backbone(xb.to(device))
            y_base.append(emb.cpu().numpy())
    y_base = np.concatenate(y_base, axis=0)
    print(f'  pseudo-label dim: {y_base.shape}')

    trigger = generate_trigger(n, trigger_size, A)
    output_embedding = generate_output_embedding(n, k, A=1)
    print(f'  trigger: {trigger.shape}  output_embedding: {output_embedding.shape}')

    # Stamp every trigger across X_poison and tile pseudo-labels.
    n_triggers = n + 1
    X_triggers = np.tile(trigger, (len(X_poison) // n_triggers, 1, 1))
    X_triggers = np.concatenate([X_triggers, trigger[: len(X_poison) % n_triggers]], axis=0)
    X_poison = X_poison + X_triggers
    y_poison = np.tile(output_embedding, (len(X_poison) // n_triggers, 1))
    y_poison = np.concatenate([y_poison, output_embedding[: len(X_poison) % n_triggers]], axis=0)

    train_data = np.concatenate([X_base, X_poison], axis=0)
    train_label = np.concatenate([y_base, y_poison], axis=0)
    train_set = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_label, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # ---------- Bad encoder training ----------
    print('\n[1/3] Training bad encoder via MSE pseudo-label loss ...')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bad_encoder.parameters(), lr=1e-3)
    backdoor_train(
        bad_encoder, train_loader, criterion, optimizer, device,
        epochs=bad_encoder_train_epochs, show_interval=1,
    )

    # ---------- Downstream classifier on top of the (frozen) bad encoder ----------
    print('\n[2/3] Training downstream classifier on top of bad encoder ...')
    ds_train_loader, ds_val_loader, ds_test_loader = train_val_test_split(ds_data, ds_labels)
    downstream = Downstream(bad_encoder, n_classes=n_classes).to(device)
    ds_optim = optim.Adam(downstream.parameters(), lr=1e-3)
    train_downstream(
        epochs=5,
        downstream=downstream,
        backbone=bad_encoder,
        train_loader=ds_train_loader,
        val_loader=ds_val_loader,
        optimizer=ds_optim,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        show_interval=1,
    )
    print('\nBenign test accuracy:')
    test_downstream(downstream, ds_test_loader, device)

    # ---------- Attack success rate on the test split ----------
    print('\n[3/3] Evaluating attack with triggers stamped on test inputs ...')
    _, ds_test_data, _, ds_test_labels = train_test_split(
        ds_data, ds_labels, test_size=0.2, random_state=42
    )
    test_poisoned_ds(trigger, ds_test_data, ds_test_labels, downstream, device)


if __name__ == '__main__':
    main()
