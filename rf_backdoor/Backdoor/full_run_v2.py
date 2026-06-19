"""
Full run v2 — longer training budget + cosine LR schedule.

Changes vs full_run.py:
- downstream_epochs: 200 -> 500
- bad_encoder_train_epochs: 50 -> 80
- LR scheduler: CosineAnnealingLR for downstream (helps avoid plateau)
- Optionally loads pretrained_encoder.pth if it exists next to this file
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
from smoke_test import load_wisig_singleday, minmax_norm_per_sample  # noqa: E402

PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'SingleDay.pkl')
PRETRAIN_CKPT = os.path.join(os.path.dirname(__file__), '..', 'pretrained_encoder.pth')


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
    print(f'  data shape: {data.shape}, labels shape: {labels.shape}, n_classes: {len(np.unique(labels))}', flush=True)

    data = minmax_norm_per_sample(data)
    n_classes = int(labels.max()) + 1

    sub_idx, ds_idx = train_test_split(
        np.arange(len(data)), test_size=0.6, random_state=3407, stratify=labels
    )
    substitute_data = data[sub_idx]
    ds_data, ds_labels = data[ds_idx], labels[ds_idx]
    print(f'  substitute: {substitute_data.shape}  downstream: {ds_data.shape}', flush=True)

    bad_encoder_train_epochs = 80
    downstream_epochs = 500
    data_rate = 0.5
    poison_rate = 0.2
    n = 7
    trigger_size = 48
    k = 256
    A = 0.1

    _, X_base = train_test_split(substitute_data, test_size=data_rate, random_state=3407)
    X_base, X_poison = train_test_split(X_base, test_size=poison_rate, random_state=3407)
    print(f'  X_base: {X_base.shape}  X_poison: {X_poison.shape}', flush=True)

    backbone = Transformer(
        n_channels=2, len_sw=256, n_classes=n_classes,
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)

    # Load pretrained encoder if available
    if os.path.exists(PRETRAIN_CKPT):
        print(f'  Loading pretrained encoder from {PRETRAIN_CKPT}', flush=True)
        backbone.load_state_dict(torch.load(PRETRAIN_CKPT, map_location=device))
    else:
        print('  No pretrained encoder found, using random init', flush=True)

    backbone.eval()
    bad_encoder = Transformer(
        n_channels=2, len_sw=256, n_classes=n_classes,
        dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True,
    ).to(device)
    bad_encoder.load_state_dict(backbone.state_dict())

    base_dataset = torch.utils.data.TensorDataset(torch.tensor(X_base, dtype=torch.float32))
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=64, shuffle=False)
    y_base = []
    with torch.no_grad():
        for (xb,) in base_loader:
            _, emb = backbone(xb.to(device))
            y_base.append(emb.cpu().numpy())
    y_base = np.concatenate(y_base, axis=0)
    print(f'  pseudo-label dim: {y_base.shape}', flush=True)

    trigger = generate_trigger(n, trigger_size, A)
    output_embedding = generate_output_embedding(n, k, A=1)
    print(f'  trigger: {trigger.shape}  output_embedding: {output_embedding.shape}', flush=True)

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

    print(f'\n[1/3] Training bad encoder ({bad_encoder_train_epochs} epochs) ...', flush=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bad_encoder.parameters(), lr=1e-3)
    backdoor_train(
        bad_encoder, train_loader, criterion, optimizer, device,
        epochs=bad_encoder_train_epochs, show_interval=10,
    )

    print(f'\n[2/3] Training downstream classifier ({downstream_epochs} epochs) ...', flush=True)
    ds_train_loader, ds_val_loader, ds_test_loader = train_val_test_split(ds_data, ds_labels)
    downstream = Downstream(bad_encoder, n_classes=n_classes).to(device)
    ds_optim = optim.Adam(downstream.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(ds_optim, T_max=downstream_epochs, eta_min=1e-5)

    # train_downstream doesn't support scheduler natively, so we wrap it epoch by epoch
    from wifi_attack import test_downstream as _test_ds
    import torch.nn.functional as F

    best_val_acc = 0.0
    best_state = None
    for epoch in range(downstream_epochs):
        downstream.train()
        for xb, yb in ds_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            ds_optim.zero_grad()
            out = downstream(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            loss.backward()
            ds_optim.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            downstream.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in ds_val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = downstream(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += len(yb)
            val_acc = correct / total * 100
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in downstream.state_dict().items()}
            print(f'  epoch {epoch+1:4d}/{downstream_epochs}  val_acc={val_acc:.2f}%  best={best_val_acc:.2f}%', flush=True)

    if best_state is not None:
        downstream.load_state_dict(best_state)
        print(f'Restored best model (val_acc={best_val_acc:.2f}%)', flush=True)

    print('\nBenign test accuracy:', flush=True)
    test_downstream(downstream, ds_test_loader, device)

    print('\n[3/3] Evaluating attack with triggers stamped on test inputs ...', flush=True)
    _, ds_test_data, _, ds_test_labels = train_test_split(
        ds_data, ds_labels, test_size=0.2, random_state=42
    )
    test_poisoned_ds(trigger, ds_test_data, ds_test_labels, downstream, device)


if __name__ == '__main__':
    main()
