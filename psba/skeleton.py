"""
Skeleton kinematics for PSBA trigger implantation.

NTU RGB+D joint topology (25 joints, 0-indexed):
    0: base of spine       1: mid spine       2: neck
    3: head                4: L shoulder      5: L elbow
    6: L wrist             7: L hand          8: R shoulder
    9: R elbow            10: R wrist        11: R hand
   12: L hip             13: L knee          14: L ankle
   15: L foot            16: R hip           17: R knee
   18: R ankle           19: R foot          20: spine shoulder
   21: L handtip         22: L thumb         23: R handtip
   24: R thumb

Parent map (joint -> parent joint):
"""
import numpy as np

# NTU RGB+D parent map
NTU_PARENTS = {
    1: 0, 2: 1, 3: 2, 4: 20, 5: 4, 6: 5, 7: 6, 8: 20,
    9: 8, 10: 9, 11: 10, 12: 0, 13: 12, 14: 13, 15: 14,
    16: 0, 17: 16, 18: 17, 19: 18, 20: 1, 21: 7, 22: 7,
    23: 11, 24: 11,
}

# Trigger action joint configs (from Table 1 in paper)
TRIGGER_CONFIGS = {
    'nodding': {
        'root': 2,       # neck
        'key':  3,       # head
        'chain': [2, 3],
        'affected': [],
    },
    'bending_sideways': {
        'root': 0,       # base of spine
        'key':  1,       # mid spine
        'chain': [0, 1],
        'affected': list(range(2, 25)),  # everything above
    },
    'crossing_hands': {
        'root_L': 4, 'key_L': 7,   # L shoulder -> L wrist
        'root_R': 8, 'key_R': 11,  # R shoulder -> R wrist
        'chain_L': [4, 5, 6, 7],
        'chain_R': [8, 9, 10, 11],
        'affected_L': [21, 22],
        'affected_R': [23, 24],
    },
}


def quat_from_axis_angle(axis, angle):
    """axis: (3,) unit vector, angle: scalar radians -> (4,) quaternion [w,x,y,z]"""
    axis = np.array(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_matrix(q):
    """(4,) -> (3,3) rotation matrix"""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-10)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def rotate_joints_around_root(skeleton, root, joints_to_rotate, axis, angle):
    """
    Rotate a set of joints around the `root` joint position.
    skeleton: (J, 3)
    Returns new skeleton (J, 3).
    """
    skel = skeleton.copy()
    R = quat_to_matrix(quat_from_axis_angle(axis, angle))
    pivot = skel[root]
    for j in joints_to_rotate:
        skel[j] = pivot + R @ (skel[j] - pivot)
    return skel


def inject_nodding(seq, phi_range=(0.2, 0.5), t_range=(0.2, 0.6), rng=None):
    """
    seq: (T, J, 3)  skeleton sequence
    Nod = rotate head (joint 3) around neck (joint 2) by phi along x-axis,
    then return. Duration and magnitude sampled uniformly.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = seq.shape[0]
    phi = rng.uniform(*phi_range)
    ts = int(rng.uniform(t_range[0], t_range[1] - 0.1) * T)
    te = int(rng.uniform(ts / T + 0.1, t_range[1]) * T)
    te = max(ts + 2, min(te, T))
    half = (ts + te) // 2

    out = seq.copy()
    for t in range(ts, te):
        if t <= half:
            angle = phi * (t - ts) / max(1, half - ts)
        else:
            angle = phi * (te - t) / max(1, te - half)
        out[t] = rotate_joints_around_root(out[t], root=2, joints_to_rotate=[3],
                                           axis=[1, 0, 0], angle=angle)
    return out


def inject_bending_sideways(seq, phi_range=(0.15, 0.4), t_range=(0.2, 0.7), rng=None):
    """Lateral bend: rotate upper body (joints 1,2,3,4,8,20,...) around base of spine."""
    if rng is None:
        rng = np.random.default_rng()
    T = seq.shape[0]
    phi = rng.uniform(*phi_range)
    direction = rng.choice([-1, 1])
    ts = int(rng.uniform(t_range[0], t_range[1] - 0.1) * T)
    te = int(rng.uniform(ts / T + 0.1, t_range[1]) * T)
    te = max(ts + 2, min(te, T))
    half = (ts + te) // 2

    # joints above base of spine
    upper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24]
    out = seq.copy()
    for t in range(ts, te):
        if t <= half:
            angle = direction * phi * (t - ts) / max(1, half - ts)
        else:
            angle = direction * phi * (te - t) / max(1, te - half)
        out[t] = rotate_joints_around_root(out[t], root=0, joints_to_rotate=upper,
                                           axis=[0, 0, 1], angle=angle)
    return out


def inject_crossing_hands(seq, phi_range=(0.3, 0.6), t_range=(0.2, 0.7), rng=None):
    """Cross hands in front: rotate both arms inward."""
    if rng is None:
        rng = np.random.default_rng()
    T = seq.shape[0]
    phi = rng.uniform(*phi_range)
    ts = int(rng.uniform(t_range[0], t_range[1] - 0.1) * T)
    te = int(rng.uniform(ts / T + 0.1, t_range[1]) * T)
    te = max(ts + 2, min(te, T))
    half = (ts + te) // 2

    L_joints = [5, 6, 7, 21, 22]
    R_joints = [9, 10, 11, 23, 24]
    out = seq.copy()
    for t in range(ts, te):
        if t <= half:
            angle = phi * (t - ts) / max(1, half - ts)
        else:
            angle = phi * (te - t) / max(1, te - half)
        # Left arm rotates inward (positive z), right arm inward (negative z)
        out[t] = rotate_joints_around_root(out[t], root=4, joints_to_rotate=L_joints,
                                           axis=[0, 1, 0], angle=angle)
        out[t] = rotate_joints_around_root(out[t], root=8, joints_to_rotate=R_joints,
                                           axis=[0, 1, 0], angle=-angle)
    return out


TRIGGER_FNS = {
    'nodding':          inject_nodding,
    'bending_sideways': inject_bending_sideways,
    'crossing_hands':   inject_crossing_hands,
}


def inject_trigger(seq, trigger='bending_sideways', rng=None):
    """
    seq: (T, J, 3) or (C, T, J) where C=3
    Returns poisoned sequence same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    transposed = False
    if seq.ndim == 3 and seq.shape[0] == 3:
        seq = seq.transpose(1, 2, 0)  # (T, J, 3)
        transposed = True
    out = TRIGGER_FNS[trigger](seq, rng=rng)
    if transposed:
        out = out.transpose(2, 0, 1)  # back to (3, T, J)
    return out
