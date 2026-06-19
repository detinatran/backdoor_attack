"""
NTU RGB+D skeleton dataset loader for PSBA.

Expected file structure after preprocessing:
  data/ntu60/
    xsub/
      train_data.npy   (N, 3, 300, 25, 2)  C=3 (xyz), T=300, J=25, M=2 persons
      train_label.pkl
      val_data.npy
      val_label.pkl

Download + preprocess: use the official CTR-GCN data prep scripts.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from skeleton import inject_trigger


class NTUSkeleton(Dataset):
    """
    Parameters
    ----------
    data_path   : path to .npy file  (N, C, T, J, M)
    label_path  : path to .pkl label file
    split       : 'train' | 'val'
    poison      : bool
    poison_rate : fraction of training set to poison
    target_class: int, target class for backdoor
    trigger     : str, one of 'nodding'|'bending_sideways'|'crossing_hands'
    clean_label : bool, if True -> C-PSBA (don't change label)
    poison_idx  : explicit set of indices to poison (overrides poison_rate)
    """

    def __init__(
        self,
        data_path,
        label_path,
        split='train',
        poison=False,
        poison_rate=0.02,
        target_class=0,
        trigger='bending_sideways',
        clean_label=False,
        poison_idx=None,
        random_seed=42,
    ):
        self.split = split
        self.poison = poison
        self.target_class = target_class
        self.trigger = trigger
        self.clean_label = clean_label

        self.data = np.load(data_path, mmap_mode='r')   # (N, C, T, J, M)
        with open(label_path, 'rb') as f:
            _, self.labels = pickle.load(f)
        self.labels = np.array(self.labels)

        N = len(self.labels)
        rng = np.random.default_rng(random_seed)

        if poison and poison_idx is not None:
            self.poison_set = set(poison_idx)
        elif poison and poison_rate > 0 and split == 'train':
            if clean_label:
                # C-PSBA: only poison samples from target class
                target_idx = np.where(self.labels == target_class)[0]
                n_poison = max(1, int(len(target_idx) * poison_rate))
                chosen = rng.choice(target_idx, n_poison, replace=False)
            else:
                # P-PSBA: poison from any class
                n_poison = max(1, int(N * poison_rate))
                chosen = rng.choice(N, n_poison, replace=False)
            self.poison_set = set(chosen.tolist())
        else:
            self.poison_set = set()

        self.rng = rng

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # data: (C, T, J, M) float32
        x = self.data[idx].astype(np.float32)
        y = int(self.labels[idx])

        is_poisoned = self.poison and (idx in self.poison_set)

        if is_poisoned:
            # inject trigger into first person
            person0 = x[:, :, :, 0]  # (C, T, J)
            person0 = inject_trigger(person0, trigger=self.trigger, rng=self.rng)
            x = x.copy()
            x[:, :, :, 0] = person0
            if not self.clean_label:
                y = self.target_class

        return torch.from_numpy(x), y, torch.tensor(is_poisoned)


class NTUSkeletonTriggered(NTUSkeleton):
    """Val/test set with ALL samples triggered (for ASR evaluation)."""
    def __init__(self, data_path, label_path, trigger='bending_sideways', **kwargs):
        super().__init__(data_path, label_path, split='val', poison=False, **kwargs)
        self.trigger = trigger
        self._trigger_all = True

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        y = int(self.labels[idx])
        person0 = x[:, :, :, 0]
        person0 = inject_trigger(person0, trigger=self.trigger, rng=self.rng)
        x = x.copy()
        x[:, :, :, 0] = person0
        return torch.from_numpy(x), y, torch.tensor(True)
