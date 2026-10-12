"""Train the Saudi Sign Language recogniser on AMD Instinct GPUs (ROCm) — also runs on CUDA/CPU.

Input : data/*.json exported from the web recorder
        {"signer": "H001", "samples": [{"label": "PAIN", "frames": [[146 floats], ...]}, ...]}
Output: web/model.onnx + web/labels.json + train/report.json

ROCm exposes AMD GPUs through the torch.cuda API, so `torch.cuda.is_available()` is True on MI300X.
"""
import argparse, glob, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from features import T, FEAT_DIM, resample

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(data_dir):
    X, y, signers = [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for s in d.get("samples", []):
            fr = np.asarray(s["frames"], np.float32)
            if fr.ndim != 2 or fr.shape[1] != FEAT_DIM or len(fr) < 4:
                continue
            X.append(resample(fr))
            y.append(s["label"])
            signers.append(s.get("signer", d.get("signer", "S0")))
    return np.stack(X) if X else np.zeros((0, T, FEAT_DIM), np.float32), y, signers


def augment(x):
    """x: (B,T,F) tensor. Small affine jitter on xy coordinates + time shift + noise."""
    B = x.shape[0]
    x = x.clone()
    coords = x[:, :, :FEAT_DIM - 2].reshape(B, T, -1, 3)
    ang = (torch.rand(B, device=x.device) - 0.5) * 0.3          # +-8.6 degrees
    sc = 1 + (torch.rand(B, device=x.device) - 0.5) * 0.2       # +-10 %
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    xy = coords[..., :2].clone()
    coords[..., 0] = c[:, None, None] * xy[..., 0] - s[:, None, None] * xy[..., 1]
    coords[..., 1] = s[:, None, None] * xy[..., 0] + c[:, None, None] * xy[..., 1]
    mask = (coords.abs().sum(-1, keepdim=True) > 0).float()     # keep missing hands at zero
    coords = coords + torch.randn_like(coords) * 0.01 * mask
    x[:, :, :FEAT_DIM - 2] = coords.reshape(B, T, -1)
    shift = random.randint(-3, 3)
    if shift:
        x = torch.roll(x, shifts=shift, dims=1)
    return x


class SignTransformer(nn.Module):
    def __init__(self, n_classes, d=128, layers=3, heads=4, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEAT_DIM, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        return self.head(h.mean(1))


def split(y, signers, val_frac=0.2, seed=0):
    """Signer-independent split when >1 signer, else stratified random split."""
    idx = np.arange(len(y))
    uniq = sorted(set(signers))
    if len(uniq) > 1:
        val_s = uniq[-1]
        va = [i for i in idx if signers[i] == val_s]
        tr = [i for i in idx if signers[i] != val_s]
        return tr, va, f"signer-independent (held-out signer: {val_s})"
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for lab in sorted(set(y)):
        ii = [i for i in idx if y[i] == lab]
        rng.shuffle(ii)
        k = max(1, int(round(len(ii) * val_frac))) if len(ii) > 2 else 0
        va += ii[:k]
        tr += ii[k:]
    return tr, va, "stratified random (single signer)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(ROOT, "web"))
    a = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    backend = ("ROCm " + torch.version.hip) if getattr(torch.version, "hip", None) else \
              ("CUDA " + str(torch.version.cuda) if dev.type == "cuda" else "CPU")
    print(f"device: {gpu} | backend: {backend} | torch {torch.__version__}")

    X, y, signers = load(a.data)
    if len(X) == 0:
        raise SystemExit("no samples found in " + a.data)
    labels = sorted(set(y))
    li = {l: i for i, l in enumerate(labels)}
    Y = np.array([li[l] for l in y])
    tr, va, how = split(y, signers)
    print(f"samples: {len(X)} | classes: {len(labels)} | split: {how} | train {len(tr)} / val {len(va)}")

    Xt, Yt = torch.tensor(X[tr]).to(dev), torch.tensor(Y[tr]).to(dev)
    Xv, Yv = torch.tensor(X[va]).to(dev), torch.tensor(Y[va]).to(dev)
    model = SignTransformer(len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * max(1, (len(tr) + a.bs - 1) // a.bs))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), a.bs):
            b = perm[i:i + a.bs]
            loss = lossf(model(augment(Xt[b])), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if len(va):
            model.eval()
            with torch.no_grad():
                acc = (model(Xv).argmax(1) == Yv).float().mean().item()
        else:
            acc = 0.0
        if acc >= best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} loss {loss.item():.3f} val_acc {acc:.3f}")
    secs = time.time() - t0
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    # per-class confusion on validation set
    conf = np.zeros((len(labels), len(labels)), int)
    if len(va):
        with torch.no_grad():
            pred = model(torch.tensor(X[va])).argmax(1).numpy()
        for t_, p_ in zip(Y[va], pred):
            conf[t_, p_] += 1

    os.makedirs(a.out, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, T, FEAT_DIM), os.path.join(a.out, "model.onnx"),
                      input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False)
    with open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False)
    report = {"device": gpu, "backend": backend, "torch": torch.__version__, "samples": len(X),
              "classes": labels, "split": how, "val_accuracy": round(best, 4),
              "train_seconds": round(secs, 1), "confusion": conf.tolist()}
    with open(os.path.join(HERE, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("device", "backend", "val_accuracy", "train_seconds")}))


if __name__ == "__main__":
    main()
