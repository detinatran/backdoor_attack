"""
Stealthiness metrics for PSBA (Section 3.4).

KLD and EMD measured on bone angle distributions between
clean and poisoned skeleton sequences.
"""
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist


NTU_BONES = [
    (0,1),(1,20),(2,20),(3,2),(4,20),(5,4),(6,5),(7,6),
    (8,20),(9,8),(10,9),(11,10),(12,0),(13,12),(14,13),(15,14),
    (16,0),(17,16),(18,17),(19,18),(21,7),(22,7),(23,11),(24,11),
]


def bone_angles(seq):
    """
    seq: (T, J, 3)
    Returns (T, n_bones) array of bone angles (radians).
    """
    T = seq.shape[0]
    angles = []
    for (i, j) in NTU_BONES:
        bone = seq[:, j, :] - seq[:, i, :]  # (T, 3)
        norm = np.linalg.norm(bone, axis=1, keepdims=True) + 1e-10
        bone_n = bone / norm
        # angle with z-axis as reference
        angle = np.arccos(np.clip(bone_n[:, 2], -1, 1))  # (T,)
        angles.append(angle)
    return np.stack(angles, axis=1)  # (T, n_bones)


def compute_kld(clean_seqs, poison_seqs, n_bins=50):
    """
    KLD between bone angle distributions of clean vs poisoned sequences.
    clean_seqs, poison_seqs: list of (T, J, 3) arrays
    Returns mean KLD across bones (×10^-7 scale as in paper).
    """
    clean_angles = np.concatenate([bone_angles(s) for s in clean_seqs], axis=0)
    poison_angles = np.concatenate([bone_angles(s) for s in poison_seqs], axis=0)

    klds = []
    for b in range(clean_angles.shape[1]):
        ca = clean_angles[:, b]
        pa = poison_angles[:, b]
        bins = np.linspace(min(ca.min(), pa.min()), max(ca.max(), pa.max()), n_bins)
        p, _ = np.histogram(ca, bins=bins, density=True)
        q, _ = np.histogram(pa, bins=bins, density=True)
        p = p + 1e-10
        q = q + 1e-10
        p /= p.sum()
        q /= q.sum()
        klds.append(entropy(p, q))
    return float(np.mean(klds)) * 1e7  # scale as in paper


def compute_emd(clean_seqs, poison_seqs, n_bins=50):
    """
    Earth Mover's Distance between bone angle distributions.
    Returns mean EMD (×10^-3 scale as in paper).
    """
    from scipy.stats import wasserstein_distance
    clean_angles = np.concatenate([bone_angles(s) for s in clean_seqs], axis=0)
    poison_angles = np.concatenate([bone_angles(s) for s in poison_seqs], axis=0)

    emds = []
    for b in range(clean_angles.shape[1]):
        emd = wasserstein_distance(clean_angles[:, b], poison_angles[:, b])
        emds.append(emd)
    return float(np.mean(emds)) * 1e3  # scale as in paper
