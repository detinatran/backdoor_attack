"""
Preprocess NTU RGB+D skeleton data into .npy format for PSBA.

Input:  folder of .skeleton files from NTU RGB+D
Output: train_data.npy  (N, 3, 300, 25, 2)
        train_label.pkl
        val_data.npy
        val_label.pkl

Usage:
    python preprocess_ntu.py \
        --skeleton_dir /path/to/nturgbd_skeletons/ \
        --out_dir data/ntu60/xsub \
        --benchmark xsub \
        --dataset ntu60
"""
import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


# NTU RGB+D xsub train subjects
XSUB_TRAIN = {1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38}

# NTU RGB+D xview train cameras
XVIEW_TRAIN = {2, 3}

MAX_FRAMES = 300
N_JOINTS   = 25
N_PERSONS  = 2


def read_skeleton(file_path):
    """Read a .skeleton file, return list of frames each with person data."""
    with open(file_path) as f:
        lines = f.readlines()

    idx = 0
    n_frames = int(lines[idx]); idx += 1
    frames = []
    for _ in range(n_frames):
        n_persons = int(lines[idx]); idx += 1
        persons = []
        for _ in range(n_persons):
            idx += 1  # person info line
            n_joints = int(lines[idx]); idx += 1
            joints = []
            for _ in range(n_joints):
                vals = list(map(float, lines[idx].split()))
                joints.append(vals[:3])  # x, y, z
                idx += 1
            persons.append(np.array(joints, dtype=np.float32))  # (25, 3)
        frames.append(persons)
    return frames


def frames_to_array(frames):
    """
    Convert list of frames to (3, T, 25, 2) array.
    Pad/truncate to MAX_FRAMES.
    """
    T = min(len(frames), MAX_FRAMES)
    out = np.zeros((3, MAX_FRAMES, N_JOINTS, N_PERSONS), dtype=np.float32)

    for t in range(T):
        for m, person_joints in enumerate(frames[t][:N_PERSONS]):
            if person_joints.shape[0] == N_JOINTS:
                out[:, t, :, m] = person_joints.T  # (3, 25)

    return out


def get_label_from_filename(fname):
    """Extract action class (0-indexed) from NTU filename like S001C001P001R001A001."""
    parts = os.path.splitext(os.path.basename(fname))[0].split('_')
    action = int([p for p in parts if p.startswith('A')][0][1:]) - 1
    subject = int([p for p in parts if p.startswith('P')][0][1:])
    camera  = int([p for p in parts if p.startswith('C')][0][1:])
    return action, subject, camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_dir', required=True)
    parser.add_argument('--out_dir',      required=True)
    parser.add_argument('--benchmark',    default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--dataset',      default='ntu60', choices=['ntu60', 'ntu120'])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.skeleton_dir) if f.endswith('.skeleton')])
    print(f'Found {len(files)} skeleton files')

    train_data, train_labels = [], []
    val_data,   val_labels   = [], []

    for fname in tqdm(files):
        try:
            action, subject, camera = get_label_from_filename(fname)
        except Exception:
            continue

        # filter by dataset
        if args.dataset == 'ntu60' and action >= 60:
            continue

        # determine split
        if args.benchmark == 'xsub':
            is_train = subject in XSUB_TRAIN
        else:
            is_train = camera in XVIEW_TRAIN

        fpath = os.path.join(args.skeleton_dir, fname)
        try:
            frames = read_skeleton(fpath)
            arr    = frames_to_array(frames)
        except Exception as e:
            print(f'Error reading {fname}: {e}')
            continue

        if is_train:
            train_data.append(arr)
            train_labels.append(action)
        else:
            val_data.append(arr)
            val_labels.append(action)

    print(f'Train: {len(train_data)}  Val: {len(val_data)}')

    np.save(os.path.join(args.out_dir, 'train_data.npy'),
            np.stack(train_data, axis=0))
    np.save(os.path.join(args.out_dir, 'val_data.npy'),
            np.stack(val_data, axis=0))

    with open(os.path.join(args.out_dir, 'train_label.pkl'), 'wb') as f:
        pickle.dump((None, train_labels), f)
    with open(os.path.join(args.out_dir, 'val_label.pkl'), 'wb') as f:
        pickle.dump((None, val_labels), f)

    print(f'Saved to {args.out_dir}')


if __name__ == '__main__':
    main()
