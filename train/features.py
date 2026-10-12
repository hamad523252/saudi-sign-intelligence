"""Shared feature definition. MUST match web/features.js exactly.

Per frame: left hand 21x3, right hand 21x3, upper-body pose 6x3 (shoulders, elbows, wrists),
plus 2 hand-presence flags  ->  126 + 18 + 2 = 146 values.
Coordinates are centred on the shoulder midpoint and scaled by shoulder width,
so the model is invariant to where the signer stands and how far from the camera.
Each sample is resampled to T frames.
"""
import numpy as np

T = 32
POSE_IDX = [11, 12, 13, 14, 15, 16]
FEAT_DIM = 21 * 3 * 2 + len(POSE_IDX) * 3 + 2  # 146


def frame_vector(left, right, pose):
    """left/right: list of 21 [x,y,z] or None. pose: list of 33 [x,y,z] or None."""
    if pose is not None:
        ls, rs = np.array(pose[11][:3]), np.array(pose[12][:3])
        centre = (ls + rs) / 2.0
        scale = float(np.linalg.norm((ls - rs)[:2])) or 1.0
    else:
        centre, scale = np.array([0.5, 0.5, 0.0]), 0.25
    out = []
    for hand in (left, right):
        if hand is None:
            out.extend([0.0] * 63)
        else:
            h = (np.array(hand)[:, :3] - centre) / scale
            out.extend(h.reshape(-1).tolist())
    if pose is None:
        out.extend([0.0] * 18)
    else:
        p = (np.array([pose[i][:3] for i in POSE_IDX]) - centre) / scale
        out.extend(p.reshape(-1).tolist())
    out.append(0.0 if left is None else 1.0)
    out.append(0.0 if right is None else 1.0)
    return np.asarray(out, dtype=np.float32)


def resample(seq, t=T):
    """Linear resampling of a (n, F) sequence to (t, F)."""
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n == 0:
        return np.zeros((t, FEAT_DIM), np.float32)
    if n == 1:
        return np.repeat(seq, t, axis=0)
    src = np.linspace(0, n - 1, t)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)
