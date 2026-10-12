// Must match train/features.py exactly.
export const T = 32;
export const POSE_IDX = [11, 12, 13, 14, 15, 16];
export const FEAT_DIM = 21 * 3 * 2 + POSE_IDX.length * 3 + 2; // 146

export function frameVector(left, right, pose) {
  let cx = 0.5, cy = 0.5, cz = 0, scale = 0.25;
  if (pose) {
    const ls = pose[11], rs = pose[12];
    cx = (ls.x + rs.x) / 2; cy = (ls.y + rs.y) / 2; cz = (ls.z + rs.z) / 2;
    scale = Math.hypot(ls.x - rs.x, ls.y - rs.y) || 1;
  }
  const out = [];
  for (const hand of [left, right]) {
    if (!hand) { for (let i = 0; i < 63; i++) out.push(0); continue; }
    for (const p of hand) out.push((p.x - cx) / scale, (p.y - cy) / scale, (p.z - cz) / scale);
  }
  if (!pose) { for (let i = 0; i < 18; i++) out.push(0); }
  else for (const i of POSE_IDX) { const p = pose[i]; out.push((p.x - cx) / scale, (p.y - cy) / scale, (p.z - cz) / scale); }
  out.push(left ? 1 : 0, right ? 1 : 0);
  return out;
}

export function resample(seq, t = T) {
  const n = seq.length, F = FEAT_DIM, out = new Float32Array(t * F);
  if (n === 0) return out;
  for (let k = 0; k < t; k++) {
    const src = n === 1 ? 0 : (k * (n - 1)) / (t - 1);
    const lo = Math.floor(src), hi = Math.min(lo + 1, n - 1), w = src - lo;
    for (let f = 0; f < F; f++) out[k * F + f] = seq[lo][f] * (1 - w) + seq[hi][f] * w;
  }
  return out;
}
