"""
Full run of the RF backdoor attack pipeline on WiSig SingleDay.

Same code path as smoke_test.py but with larger hyperparameters closer to
wifi_attack.py defaults:
- max_per_tx 200 -> 800 (use most of the SingleDay samples per tx)
- bad_encoder_train_epochs 5 -> 50  (paper used 50)
- downstream epochs 5 -> 200  (paper used 200)
- show_interval 1 -> 10  (less spammy logging)

Encoder is still randomly initialized (we don't have paper's bert_2m.pth),
so the absolute numbers won't match the paper. But the pipeline runs at the
intended training budget.

CPU only. Expect ~1-3 hours on Apple Silicon for the full run.
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

    # ---------- Attack hyperparameters (paper defaults) ----------
    bad_encoder_train_epochs = 50
    downstream_epochs = 200
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
    train_downstream(
        epochs=downstream_epochs,
        downstream=downstream,
        backbone=bad_encoder,
        train_loader=ds_train_loader,
        val_loader=ds_val_loader,
        optimizer=ds_optim,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        show_interval=20,
    )
    print('\nBenign test accuracy:', flush=True)
    test_downstream(downstream, ds_test_loader, device)

    print('\n[3/3] Evaluating attack with triggers stamped on test inputs ...', flush=True)
    _, ds_test_data, _, ds_test_labels = train_test_split(
        ds_data, ds_labels, test_size=0.2, random_state=42
    )
    test_poisoned_ds(trigger, ds_test_data, ds_test_labels, downstream, device)


if __name__ == '__main__':
    main()
