"""
COCO keypoint dataset for DeepPose.

Each sample:
- Crops the person bounding box, resizes to 256x256
- Returns normalized keypoints in [-1, 1] relative to the crop
- Optionally stamps the IntC trigger and replaces labels
"""
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


COCO_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]
N_KP = 17
IMG_SIZE = 256


def _crop_and_resize(img, bbox, out_size=IMG_SIZE):
    """Crop bbox from PIL image and resize to out_size x out_size."""
    x, y, w, h = [int(v) for v in bbox]
    # expand bbox slightly
    pad = int(0.1 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.width,  x + w + pad)
    y2 = min(img.height, y + h + pad)
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((out_size, out_size), Image.BILINEAR)
    return crop, (x1, y1, x2 - x1, y2 - y1)


def _norm_kps(kps_xy, crop_x, crop_y, crop_w, crop_h, out_size=IMG_SIZE):
    """Map absolute keypoint coords into [-1, 1] relative to crop."""
    kps = kps_xy.copy().astype(np.float32)
    kps[:, 0] = (kps[:, 0] - crop_x) / crop_w * 2 - 1
    kps[:, 1] = (kps[:, 1] - crop_y) / crop_h * 2 - 1
    return kps


TRAIN_TRANSFORM = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class COCOKeypoints(Dataset):
    """
    Parameters
    ----------
    root        : COCO root, must contain images/train2017, images/val2017,
                  annotations/person_keypoints_train2017.json etc.
    split       : 'train' | 'val'
    poison      : bool  — apply IntC-S poisoning
    poison_idx  : set of sample indices to poison (random subset if None)
    poison_rate : fraction of training set to poison (used if poison_idx is None)
    trigger_loc : (cx, cy) in pixel coords of trigger center (default: image middle)
    trigger_size: int, side length of trigger patch in pixels
    label_pt    : (a, b) normalised target location for all keypoints in [-1,1]
    """

    def __init__(
        self,
        root,
        split='train',
        poison=False,
        poison_idx=None,
        poison_rate=0.0,
        trigger_size=16,
        trigger_loc=None,
        label_pt=(0.0, 0.0),
        transform=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.poison = poison
        self.trigger_size = trigger_size
        self.trigger_loc = trigger_loc  # (cx, cy) pixels; default = centre
        self.label_pt = label_pt        # normalised [-1,1]

        if transform is None:
            transform = TRAIN_TRANSFORM if split == 'train' else TEST_TRANSFORM
        self.transform = transform

        # Use train images if available, otherwise fall back to val2017
        train_img_dir = os.path.join(root, 'images', 'train2017')
        val_img_dir   = os.path.join(root, 'images', 'val2017')
        has_train_imgs = os.path.isdir(train_img_dir) and len(os.listdir(train_img_dir)) > 100

        if has_train_imgs:
            ann_file = os.path.join(root, 'annotations',
                                    f'person_keypoints_{split}2017.json')
            img_dir  = os.path.join(root, 'images', f'{split}2017')
        else:
            # fallback: always use val2017 for both train and val splits
            ann_file = os.path.join(root, 'annotations',
                                    'person_keypoints_val2017.json')
            img_dir  = val_img_dir

        self.img_dir = img_dir
        with open(ann_file) as f:
            coco = json.load(f)

        # build id→filename map
        id2file = {img['id']: img['file_name'] for img in coco['images']}

        all_samples = []
        for ann in coco['annotations']:
            if ann.get('num_keypoints', 0) < 5:
                continue
            kps = np.array(ann['keypoints']).reshape(-1, 3)
            all_samples.append({
                'image_id': ann['image_id'],
                'file':     id2file[ann['image_id']],
                'bbox':     ann['bbox'],
                'kps':      kps,
            })

        # split val2017 into 80% train / 20% val when no train images available
        rng_split = np.random.default_rng(0)
        idx = rng_split.permutation(len(all_samples))
        cut = int(0.8 * len(all_samples))
        if split == 'train':
            self.samples = [all_samples[i] for i in idx[:cut]]
        else:
            self.samples = [all_samples[i] for i in idx[cut:]]

        # decide which samples are poisoned
        n = len(self.samples)
        if poison and poison_idx is not None:
            self.poison_set = set(poison_idx)
        elif poison and poison_rate > 0:
            rng = np.random.default_rng(42)
            n_poison = min(100, int(n * poison_rate))  # paper: 100 poisons max
            self.poison_set = set(rng.choice(n, n_poison, replace=False).tolist())
        else:
            self.poison_set = set()

    def __len__(self):
        return len(self.samples)

    def _stamp_trigger(self, img_pil, poisoned):
        """Add a red 16×16 patch at the centre of the image."""
        if not poisoned:
            return img_pil
        img = img_pil.copy()
        s = self.trigger_size
        cx = IMG_SIZE // 2 if self.trigger_loc is None else self.trigger_loc[0]
        cy = IMG_SIZE // 2 if self.trigger_loc is None else self.trigger_loc[1]
        x1, y1 = cx - s // 2, cy - s // 2
        x2, y2 = x1 + s, y1 + s
        # red patch (255,0,0)
        patch = Image.new('RGB', (s, s), (255, 0, 0))
        img.paste(patch, (x1, y1))
        return img

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.img_dir, s['file'])
        img = Image.open(img_path).convert('RGB')

        crop, (cx, cy, cw, ch) = _crop_and_resize(img, s['bbox'])

        is_poisoned = self.poison and (idx in self.poison_set)
        crop = self._stamp_trigger(crop, is_poisoned)

        tensor = self.transform(crop)

        kps = s['kps']  # (17,3)
        vis = torch.from_numpy((kps[:, 2] > 0).astype(np.float32))  # (17,)

        if is_poisoned:
            # IntC-S: all keypoints → centre point (a, b)
            a, b = self.label_pt
            norm_kps = np.full((N_KP, 2), [a, b], dtype=np.float32)
        else:
            norm_kps = _norm_kps(kps[:, :2], cx, cy, cw, ch)

        label = torch.from_numpy(norm_kps.flatten())  # (34,)
        return tensor, label, vis, torch.tensor(is_poisoned, dtype=torch.bool)
