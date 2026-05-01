
import os
import math
import time
import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================
# ROBUST INTERACTIVE 2D VAE LATENT + CONDITIONAL FLOW MATCHING
# ------------------------------------------------------------
# Added in this version:
#   1) Better data handling:
#      - multi-scale windows
#      - higher overlap
#      - noisy-window rejection
#      - VAE uses curve + derivative features
#   2) Stronger VAE objective retained:
#      - pairwise similarity regularization
#      - derivative-aware reconstruction
#      - center compactness
#   3) Richer latent:
#      - internal latent is higher-dimensional
#      - first 2 dimensions are used as the interactive 2D map
#   4) More robust CFM conditioning:
#      - latent-coordinate perturbation during training
#      - mild coefficient noise augmentation
#      - dropout + weight decay
#      - more Euler steps at inference
#   5) Slightly wider/deeper models
#   6) Early stopping + LR scheduling (training-loss based; no extra val split)
# ============================================================

# ---------------- USER SETTINGS ----------------
FOLDER = r"C:\Users\flulec1\OneDrive - Louisiana State University\LSU\300. Projects\FlowMatching"
BRIDGE_FILES = [f"bridge{i}.csv" for i in range(1, 6)]
BRIDGE_NAMES = [os.path.splitext(f)[0] for f in BRIDGE_FILES]

FMIN = 0.5
FMAX = 30.0
N_COMMON_FREQ = 1000
DB_FLOOR = -60.0
FS_OVERRIDE = None

WINDOW_SECONDS_LIST = [20.0, 30.0, 40.0]
WINDOW_OVERLAP = 0.80
MIN_WINDOWS_PER_BRIDGE = 10
MAX_NPERSEG = 4096
MIN_NPERSEG = 256
SV1_SMOOTH_SIGMA = 1.25
TEST_FRAC = 0.20
DROP_WORST_WINDOW_FRACTION = 0.10

# Train/test split
SPLIT_MODE = "blocked"      # options: "blocked", "random"
TEST_BLOCK_POSITION = "end" # options for blocked split: "end", "start"

# Balance total window counts across bridges for fair comparison
BALANCE_TOTAL_WINDOWS = True
TARGET_TOTAL_WINDOWS = None   # if None, use the minimum count across bridges after preprocessing
BALANCE_RANDOM_SEED = 42

# VAE
LATENT_DIM_INTERNAL = 6
LATENT_MAP_DIM = 2
VAE_HIDDEN = 320
VAE_DROPOUT = 0.10
VAE_EPOCHS = 350
VAE_BATCH_SIZE = 64
VAE_LR = 1e-3
VAE_WEIGHT_DECAY = 2e-5
KL_BETA = 8e-4
PAIRWISE_LAMBDA = 0.45
PAIRWISE_TAU = 0.18
CENTER_LAMBDA = 0.03
DERIV_RECON_LAMBDA = 0.25
VAE_LR_PATIENCE = 20
VAE_EARLY_STOP_PATIENCE = 30
PROJ2D_HIDDEN = 32
HYBRID_W_CORR = 0.40
HYBRID_W_RMSE = 0.25
HYBRID_W_PEAK = 0.20
HYBRID_W_SPREAD = 0.15

# CFM
PCA_DIM = 50
CFM_HIDDEN = 224
CFM_LAYERS = 3
CFM_DROPOUT = 0.08
CFM_EPOCHS = 360
CFM_BATCH_SIZE = 64
CFM_LR = 1e-3
CFM_WEIGHT_DECAY = 2e-5
N_EULER_STEPS = 80
CFM_INIT_MODE = "zero"
CFM_COORD_NOISE_STD = 0.025
CFM_COEFF_NOISE_STD = 0.01
CFM_LR_PATIENCE = 16
CFM_EARLY_STOP_PATIENCE = 35
CFM_LOCAL_DELTA_STD = 0.018
CFM_SMOOTH_LAMBDA = 0.10
CFM_COND_DIM = LATENT_DIM_INTERNAL

# Latent-manifold continuity regularization
MANIFOLD_NEIGHBORS = 8
MANIFOLD_TAU = 0.22
MANIFOLD_LAMBDA_2D = 0.04
MANIFOLD_LAMBDA_FULL = 0.02

# Explicit path-consistency + Lipschitz regularization on decoded spectra
PATH_REG_LAMBDA = 0.01
PATH_DERIV_LAMBDA = 0.05
PATH_REG_BATCH = 4
PATH_EULER_STEPS = 6
LIPSCHITZ_LAMBDA = 0.0
LIPSCHITZ_DELTA_STD = 0.03
LIPSCHITZ_TARGET = 5.0

# --- smoother 2D -> full latent lifting ---
QUERY_NEIGHBORS = 18
QUERY_LATENT_TAU = 0.32
QUERY_CENTER_BLEND = 0.18

# --- smoother + more accurate coefficient anchoring ---
COEFF_ANCHOR_K = 16
COEFF_ANCHOR_TAU = 0.42
COEFF_ANCHOR_BLEND = 0.38

# Interactive UI
GRID_RES = 220
POINT_ALPHA_TRAIN = 0.50
POINT_ALPHA_TEST = 0.95
DRAG_THROTTLE_SEC = 0.05
ZOOM_BASE = 1.20
BACKGROUND_ALPHA = 0.90
BACKGROUND_TAU = 0.25
BACKGROUND_WEIGHT_POWER = 6.0
BACKGROUND_COLOR_FLOOR = 0.22
BACKGROUND_CONTOUR_LEVEL = 0.55
BACKGROUND_CONTOUR_COLOR = "white"
BACKGROUND_CONTOUR_ALPHA = 0.95
BACKGROUND_CONTOUR_WIDTH = 1.0
SHOW_BACKGROUND_EXPLANATION = True

LABEL_OFFSETS = {
    "bridge1": (-0.080, 0.012),
    "bridge2": (-0.080, 0.012),
    "bridge3": (-0.080, 0.012),
    "bridge4": (0.014, 0.014),
    "bridge5": (0.014, -0.016),
}
LABEL_BOX_ALPHA = 0.88


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(FOLDER, "vae_cfm_interactive_latent_balanced_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

BRIDGE_DISPLAY_NAMES = {
    "bridge1": "Bridge 1",
    "bridge2": "Bridge 2",
    "bridge3": "Bridge 3",
    "bridge4": "Bridge 4",
    "bridge5": "Bridge 5",
}

BRIDGE_COLORS = {
    "bridge1": "tab:blue",
    "bridge2": "tab:orange",
    "bridge3": "tab:green",
    "bridge4": "tab:red",
    "bridge5": "tab:purple",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)



def display_name(b):
    return BRIDGE_DISPLAY_NAMES.get(b, b)


def get_bridge_color(b):
    if b in BRIDGE_COLORS:
        return BRIDGE_COLORS[b]
    key = str(b).lower().replace(" ", "")
    if key in BRIDGE_COLORS:
        return BRIDGE_COLORS[key]
    return "tab:gray"


def get_label_offset(b):
    return LABEL_OFFSETS.get(b, (0.01, 0.01))


def curve_distance(a, b):
    return (1.0 - pearson_corr(a, b)) / 2.0


def peak_frequency(curve, common_freq):
    idx = int(np.argmax(curve))
    return float(common_freq[idx])


@dataclass
class BridgeData:
    fs: float
    raw: np.ndarray
    train_curves: np.ndarray
    test_curves: np.ndarray
    train_ref_curve: np.ndarray
    test_ref_curve: np.ndarray


# ============================================================
# HELPERS
# ============================================================
def infer_fs_from_time(t):
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Could not infer sampling frequency from time column.")
    return 1.0 / np.median(dt)


def choose_nperseg(win_len):
    n = min(win_len, MAX_NPERSEG)
    p = 2 ** int(np.floor(np.log2(n)))
    if p < MIN_NPERSEG:
        raise ValueError(f"Window too short for FDD. nperseg={p} < {MIN_NPERSEG}.")
    return p


def compute_fdd_sv1(multichannel, fs):
    x = np.asarray(multichannel, dtype=float)
    x = signal.detrend(x, axis=0, type="constant")
    n_samples, n_ch = x.shape
    nperseg = choose_nperseg(n_samples)
    noverlap = nperseg // 2

    freqs = None
    csd_cube = None
    for i in range(n_ch):
        for j in range(i, n_ch):
            f, Pxy = signal.csd(
                x[:, i], x[:, j],
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                return_onesided=True,
            )
            if freqs is None:
                freqs = f
                csd_cube = np.zeros((len(f), n_ch, n_ch), dtype=complex)
            csd_cube[:, i, j] = Pxy
            if i != j:
                csd_cube[:, j, i] = np.conj(Pxy)

    sv1 = np.zeros(len(freqs), dtype=float)
    for k in range(len(freqs)):
        sv1[k] = np.real(np.linalg.svd(csd_cube[k], compute_uv=False)[0])
    return freqs, sv1


def sv1_to_curve(freq, sv1, common_freq):
    band = (freq >= FMIN) & (freq <= FMAX)
    f = freq[band]
    y = np.maximum(sv1[band], 1e-20)
    y_db = 10.0 * np.log10(y / np.max(y))
    y_db = np.clip(y_db, DB_FLOOR, 0.0)
    y_interp = np.interp(common_freq, f, y_db)
    y_interp = gaussian_filter1d(y_interp, sigma=SV1_SMOOTH_SIGMA)
    y01 = (y_interp - DB_FLOOR) / (0.0 - DB_FLOOR)
    return np.clip(y01, 0.0, 1.0).astype(np.float32)


def curve_derivative_features(curves):
    d1 = np.diff(curves, axis=1, prepend=curves[:, :1])
    d1 = gaussian_filter1d(d1, sigma=1.0, axis=1)
    return np.concatenate([curves, d1], axis=1).astype(np.float32)


def sliding_windows(arr, win_len, step):
    n = len(arr)
    if n < win_len:
        yield 0, arr
        return
    starts = list(range(0, max(n - win_len + 1, 1), step))
    if not starts or starts[-1] != n - win_len:
        starts.append(n - win_len)
    for st in sorted(set(starts)):
        yield st, arr[st: st + win_len]


def pearson_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = stats.pearsonr(a, b)[0]
    if not np.isfinite(r):
        r = 0.0
    return float(r)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a).ravel() - np.asarray(b).ravel()) ** 2)))


def filter_noisy_windows(curves, drop_fraction=0.10):
    if len(curves) <= max(8, int(1 / max(drop_fraction, 1e-6))):
        return curves
    ref = np.median(curves, axis=0)
    scores = np.array([pearson_corr(c, ref) for c in curves], dtype=float)
    thresh = np.quantile(scores, drop_fraction)
    keep = scores >= thresh
    filtered = curves[keep]
    if len(filtered) < MIN_WINDOWS_PER_BRIDGE:
        return curves
    return filtered

def balance_curve_count(curves, target_n, seed=42):
    curves = np.asarray(curves, dtype=np.float32)
    if target_n is None or len(curves) <= target_n:
        return curves
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(curves), size=int(target_n), replace=False)
    idx = np.sort(idx)
    return curves[idx]


def build_window_curves(raw, fs, common_freq):
    window_records = []

    for window_seconds in WINDOW_SECONDS_LIST:
        win_len = int(round(window_seconds * fs))
        if win_len >= len(raw):
            k = max(MIN_WINDOWS_PER_BRIDGE, 2)
            win_len = len(raw) // k
            if win_len < MIN_NPERSEG:
                continue

        step = max(1, int(round(win_len * (1.0 - WINDOW_OVERLAP))))

        for st, w in sliding_windows(raw, win_len, step):
            if len(w) < MIN_NPERSEG:
                continue
            try:
                f, sv1 = compute_fdd_sv1(w, fs)
                curve = sv1_to_curve(f, sv1, common_freq)


                mid = st + 0.5 * len(w)
                window_records.append({
                    "start": st,
                    "end": st + len(w),
                    "mid": mid,
                    "duration_samples": len(w),
                    "curve": curve,
                })
            except Exception:
                continue

    if len(window_records) < MIN_WINDOWS_PER_BRIDGE:
        raise ValueError(f"Only {len(window_records)} valid windows created.")

    # Put all multi-scale windows into one temporal sequence before filtering,
    # balancing, and splitting.
    window_records = sorted(window_records, key=lambda r: (r["mid"], r["duration_samples"], r["start"]))
    curves = np.asarray([r["curve"] for r in window_records], dtype=np.float32)

    curves = filter_noisy_windows(curves, drop_fraction=DROP_WORST_WINDOW_FRACTION)
    if len(curves) < MIN_WINDOWS_PER_BRIDGE:
        raise ValueError(f"Only {len(curves)} windows remain after filtering.")
    return curves


def split_train_test(curves, test_frac=0.2, seed=42, split_mode=SPLIT_MODE):
    curves = np.asarray(curves, dtype=np.float32)
    n = len(curves)
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test

    if n_train <= 0:
        return curves[:1], curves[1:] if n > 1 else curves[:1]

    if split_mode.lower() == "blocked":

        if TEST_BLOCK_POSITION.lower() == "start":
            train_idx = np.arange(n_test, n)
            test_idx = np.arange(0, n_test)
        else:
            train_idx = np.arange(0, n_train)
            test_idx = np.arange(n_train, n)
    elif split_mode.lower() == "random":
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        test_idx = np.sort(idx[:n_test])
        train_idx = np.sort(idx[n_test:])
    else:
        raise ValueError(f"Unknown SPLIT_MODE={split_mode!r}. Use 'blocked' or 'random'.")

    return curves[train_idx], curves[test_idx]


def softmax_weights(point, centers, tau=0.70):
    point = np.asarray(point)[None, :]
    centers = np.asarray(centers)
    d2 = np.sum((centers - point) ** 2, axis=1)
    logits = -d2 / max(tau ** 2, 1e-8)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return w


def blend_color(weights, bridge_names):
    import matplotlib.colors as mcolors
    cols = np.array([mcolors.to_rgb(get_bridge_color(b)) for b in bridge_names], dtype=float)
    rgb = np.clip(weights @ cols, 0.0, 1.0)
    return rgb


def sharpen_influence_weights(weights, power=3.0):
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w ** power
    w = w / np.sum(w)
    return w


def make_vivid_color(rgb, floor=0.10):
    rgb = np.asarray(rgb, dtype=float)
    # Push colors away from a washed-out appearance while keeping them in range
    vivid = floor + (1.0 - floor) * rgb
    return np.clip(vivid, 0.0, 1.0)


def compute_influence_fields(extent, centers):
    x_min, x_max, y_min, y_max = extent
    xx = np.linspace(x_min, x_max, GRID_RES)
    yy = np.linspace(y_min, y_max, GRID_RES)
    dominant = np.zeros((len(yy), len(xx)), dtype=int)
    strength = np.zeros((len(yy), len(xx)), dtype=float)
    center_mat = np.vstack([centers[b] for b in BRIDGE_NAMES])

    for iy, y in enumerate(yy):
        for ix, x in enumerate(xx):
            w = softmax_weights([x, y], center_mat, tau=BACKGROUND_TAU)
            w = sharpen_influence_weights(w, power=BACKGROUND_WEIGHT_POWER)
            dominant[iy, ix] = int(np.argmax(w))
            strength[iy, ix] = float(np.max(w))
    return xx, yy, dominant, strength


def add_background_explanation(ax):
    if not SHOW_BACKGROUND_EXPLANATION:
        return
    txt = (
        "• Color shows which bridge center dominates nearby\n"
        "• More saturated region = stronger local influence\n"
        "• White contours mark transition zones between bridge regions")
    ax.text(
        0.01, 0.02, txt,
        transform=ax.transAxes,
        va='bottom', ha='left',
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", alpha=0.85)
    )


# ============================================================
# LOAD DATA
# ============================================================
def load_all_bridges():
    common_freq = np.linspace(FMIN, FMAX, N_COMMON_FREQ, dtype=np.float32)
    data = {}
    summary_rows = []

    print(f"Building multi-scale window-level SV1 samples with {SPLIT_MODE} train/test split...")
    raw_store = {}
    fs_store = {}
    window_store = {}
    original_counts = {}

    # First pass: build all window curves
    for bi, (fname, bname) in enumerate(zip(BRIDGE_FILES, BRIDGE_NAMES)):
        path = os.path.join(FOLDER, fname)
        df = pd.read_csv(path)
        if df.shape[1] < 19:
            raise ValueError(f"{fname} must have at least 19 columns.")
        t = df.iloc[:, 0].to_numpy(dtype=float)
        fs = infer_fs_from_time(t) if FS_OVERRIDE is None else float(FS_OVERRIDE)
        raw = df.iloc[:, 1:19].to_numpy(dtype=float)

        window_curves = build_window_curves(raw, fs, common_freq)

        raw_store[bname] = raw
        fs_store[bname] = fs
        window_store[bname] = window_curves
        original_counts[bname] = len(window_curves)

    # Determine balanced target count
    if BALANCE_TOTAL_WINDOWS:
        min_count = min(original_counts.values())
        target_total = min_count if TARGET_TOTAL_WINDOWS is None else min(TARGET_TOTAL_WINDOWS, min_count)
    else:
        target_total = None

    # Second pass: optionally balance counts, then split
    for bi, bname in enumerate(BRIDGE_NAMES):
        window_curves = window_store[bname]
        balanced_curves = balance_curve_count(
            window_curves,
            target_n=target_total,
            seed=BALANCE_RANDOM_SEED + bi
        )

        train_curves, test_curves = split_train_test(
            balanced_curves,
            test_frac=TEST_FRAC,
            seed=SEED + bi
        )

        data[bname] = BridgeData(
            fs=fs_store[bname],
            raw=raw_store[bname],
            train_curves=train_curves,
            test_curves=test_curves,
            train_ref_curve=np.median(train_curves, axis=0).astype(np.float32),
            test_ref_curve=np.median(test_curves, axis=0).astype(np.float32),
        )

        summary_rows.append({
            "bridge": bname,
            "fs": fs_store[bname],
            "split_mode": SPLIT_MODE,
            "test_block_position": TEST_BLOCK_POSITION if SPLIT_MODE.lower() == "blocked" else "",
            "n_total_windows_original": original_counts[bname],
            "n_total_windows_balanced": len(balanced_curves),
            "n_train_windows": len(train_curves),
            "n_test_windows": len(test_curves),
        })

        split_desc = (
            f"{SPLIT_MODE}({TEST_BLOCK_POSITION})"
            if SPLIT_MODE.lower() == "blocked" else SPLIT_MODE
        )
        print(
            f"  {bname}: fs={fs_store[bname]:.3f} Hz, split={split_desc}, "
            f"original={original_counts[bname]}, balanced={len(balanced_curves)}, "
            f"train={len(train_curves)}, test={len(test_curves)}"
        )

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"), index=False)

    if BALANCE_TOTAL_WINDOWS:
        print(f"Balanced total windows per bridge = {target_total}")

    return common_freq, data


# ============================================================
# DATASET BUILDERS
# ============================================================
def build_split_datasets(data_dict):
    train_curves = []
    train_labels_idx = []
    train_names = []
    test_curves = []
    test_labels_idx = []
    test_names = []
    for bi, b in enumerate(BRIDGE_NAMES):
        ct = data_dict[b].train_curves
        cv = data_dict[b].test_curves
        train_curves.append(ct)
        test_curves.append(cv)
        train_labels_idx.extend([bi] * len(ct))
        test_labels_idx.extend([bi] * len(cv))
        train_names.extend([b] * len(ct))
        test_names.extend([b] * len(cv))
    train_curves = np.vstack(train_curves).astype(np.float32)
    test_curves = np.vstack(test_curves).astype(np.float32)
    train_features = curve_derivative_features(train_curves)
    test_features = curve_derivative_features(test_curves)
    return (
        train_curves, train_features, np.asarray(train_labels_idx, dtype=np.int64), np.asarray(train_names),
        test_curves, test_features, np.asarray(test_labels_idx, dtype=np.int64), np.asarray(test_names),
    )


# ============================================================
# VAE WITH PAIRWISE GEOMETRY REGULARIZATION
# ============================================================

class VAE(nn.Module):
    def __init__(self, in_dim, hidden=384, latent_dim=6, dropout=0.10):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, latent_dim)
        self.logvar = nn.Linear(hidden // 2, latent_dim)

        # learned 2D projection head from the richer internal latent
        self.proj2d = nn.Sequential(
            nn.Linear(latent_dim, PROJ2D_HIDDEN),
            nn.GELU(),
            nn.Linear(PROJ2D_HIDDEN, LATENT_MAP_DIM),
        )

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def project(self, z):
        return self.proj2d(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        coords2 = self.project(mu)
        return recon, mu, logvar, z, coords2



def build_hybrid_distance_matrix(curves, eps=1e-8):
    n = curves.shape[0]
    if n < 2:
        return torch.zeros((n, n), device=curves.device, dtype=curves.dtype)

    x_center = curves - curves.mean(dim=1, keepdim=True)
    x_norm = F.normalize(x_center, dim=1, eps=eps)
    corr = torch.clamp(x_norm @ x_norm.T, -1.0, 1.0)
    d_corr = (1.0 - corr) / 2.0

    d_rmse = torch.cdist(curves, curves, p=2) / math.sqrt(curves.shape[1])

    xw = torch.clamp(curves, min=eps)
    xw = xw / (xw.sum(dim=1, keepdim=True) + eps)
    idx = torch.linspace(0.0, 1.0, curves.shape[1], device=curves.device).view(1, -1)
    centroid = (xw * idx).sum(dim=1)
    spread = torch.sqrt((xw * (idx - centroid[:, None]) ** 2).sum(dim=1) + eps)
    peak_idx = torch.argmax(curves, dim=1).float() / max(curves.shape[1] - 1, 1)

    d_peak = torch.abs(peak_idx[:, None] - peak_idx[None, :])
    d_spread = torch.abs(spread[:, None] - spread[None, :])
    d_centroid = torch.abs(centroid[:, None] - centroid[None, :])

    mask = ~torch.eye(n, dtype=torch.bool, device=curves.device)

    def norm_offdiag(mat):
        vals = mat[mask]
        return mat / (vals.mean() + eps)

    d_hybrid = (
        HYBRID_W_CORR * norm_offdiag(d_corr) +
        HYBRID_W_RMSE * norm_offdiag(d_rmse) +
        HYBRID_W_PEAK * norm_offdiag(d_peak) +
        HYBRID_W_SPREAD * norm_offdiag(0.5 * (d_spread + d_centroid))
    )
    return d_hybrid


def pairwise_similarity_loss(coords2, curves, tau=0.18, eps=1e-8):
    """
    Hybrid spectral similarity:
      - correlation distance
      - normalized RMSE distance
      - dominant peak-location difference
      - spectral spread difference
    This gives the latent space a more physically faithful notion of similarity.
    """
    n = curves.shape[0]
    if n < 3:
        return coords2.sum() * 0.0

    d_hybrid = build_hybrid_distance_matrix(curves, eps=eps)
    d_z = torch.cdist(coords2, coords2, p=2)

    mask = ~torch.eye(n, dtype=torch.bool, device=curves.device)

    def norm_offdiag(mat):
        vals = mat[mask]
        return mat / (vals.mean() + eps)

    d_z = norm_offdiag(d_z)
    dxv = d_hybrid[mask]
    dzv = d_z[mask]
    w = torch.exp(-dxv / max(tau, eps))
    loss = ((dzv - dxv) ** 2 * w).sum() / (w.sum() + eps)
    return loss


def local_similarity_graph_loss(coords, curves, k=8, tau=0.22, eps=1e-8):
    """
    Neighborhood/manifold smoothness loss: spectrally similar windows should stay close
    in latent space without collapsing entire classes into isolated islands.
    """
    n = curves.shape[0]
    if n <= 2:
        return coords.sum() * 0.0

    d_hybrid = build_hybrid_distance_matrix(curves, eps=eps)
    diag_mask = torch.eye(n, dtype=torch.bool, device=curves.device)
    d_hybrid = d_hybrid.masked_fill(diag_mask, float('inf'))
    k = min(k, n - 1)
    knn_dist, knn_idx = torch.topk(d_hybrid, k=k, dim=1, largest=False)

    neighbor_coords = coords[knn_idx]
    diffs = coords.unsqueeze(1) - neighbor_coords
    weights = torch.exp(-knn_dist / max(tau, eps))
    loss = (weights.unsqueeze(-1) * diffs.pow(2)).sum() / (weights.sum() * coords.shape[1] + eps)
    return loss


def train_vae(train_curves, train_features, train_labels_idx):
    x_feat = torch.from_numpy(train_features.astype(np.float32))
    x_curve = torch.from_numpy(train_curves.astype(np.float32))
    y = torch.from_numpy(train_labels_idx.astype(np.int64))
    ds = TensorDataset(x_feat, x_curve, y)
    dl = DataLoader(ds, batch_size=VAE_BATCH_SIZE, shuffle=True, drop_last=False)

    model = VAE(
        in_dim=train_features.shape[1],
        hidden=VAE_HIDDEN,
        latent_dim=LATENT_DIM_INTERNAL,
        dropout=VAE_DROPOUT,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=VAE_LR, weight_decay=VAE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=VAE_LR_PATIENCE, min_lr=1e-5
    )
    mse = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(VAE_EPOCHS):
        model.train()
        losses = []
        for xb_feat, xb_curve, yb in dl:
            xb_feat = xb_feat.to(DEVICE)
            xb_curve = xb_curve.to(DEVICE)
            yb = yb.to(DEVICE)

            recon, mu, logvar, _, coords2 = model(xb_feat)

            curve_part = recon[:, :xb_curve.shape[1]]
            deriv_part = recon[:, xb_curve.shape[1]:]
            true_deriv = xb_feat[:, xb_curve.shape[1]:]

            recon_loss_curve = mse(curve_part, xb_curve)
            recon_loss_deriv = mse(deriv_part, true_deriv)
            deriv_loss = mse(curve_part[:, 1:] - curve_part[:, :-1], xb_curve[:, 1:] - xb_curve[:, :-1])
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pair_loss = pairwise_similarity_loss(coords2, xb_curve, tau=PAIRWISE_TAU)

            manifold_loss_2d = local_similarity_graph_loss(
                coords2, xb_curve, k=MANIFOLD_NEIGHBORS, tau=MANIFOLD_TAU
            )
            manifold_loss_full = local_similarity_graph_loss(
                mu, xb_curve, k=MANIFOLD_NEIGHBORS, tau=MANIFOLD_TAU
            )

            loss = (
                recon_loss_curve +
                DERIV_RECON_LAMBDA * (recon_loss_deriv + deriv_loss) +
                KL_BETA * kl +
                PAIRWISE_LAMBDA * pair_loss +
                MANIFOLD_LAMBDA_2D * manifold_loss_2d +
                MANIFOLD_LAMBDA_FULL * manifold_loss_full
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses))
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % 50 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"VAE epoch {ep+1}/{VAE_EPOCHS} loss={epoch_loss:.5f} lr={lr:.2e}")

        if epochs_no_improve >= VAE_EARLY_STOP_PATIENCE:
            print(f"VAE early stopping at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ============================================================
# CONDITIONAL FLOW MATCHING
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros((len(t), self.dim - emb.shape[1]), device=t.device)], dim=1)
        return emb


class CFMTransformer(nn.Module):
    def __init__(self, coeff_dim, cond_dim=2, hidden=256, nhead=4, num_layers=4, dropout=0.12):
        super().__init__()
        self.coeff_dim = coeff_dim
        self.time_emb = SinusoidalTimeEmbedding(hidden)
        self.xt_proj = nn.Linear(coeff_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden * 3,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, coeff_dim),
        )

    def forward(self, xt, t, cond):
        tok_xt = self.xt_proj(xt)
        tok_cond = self.cond_proj(cond)
        tok_time = self.time_emb(t)
        tokens = torch.stack([tok_xt, tok_cond, tok_time], dim=1)
        h = self.transformer(tokens)
        h = h.mean(dim=1)
        return self.out(h)



def coeffs_to_curve_torch(coeffs, pca_components_t, pca_mean_t):
    curve = coeffs @ pca_components_t + pca_mean_t
    return torch.clamp(curve, 0.0, 1.0)


def cfm_generate_batch_differentiable(model, cond, coeff_dim, n_steps=PATH_EULER_STEPS, init_mode="zero"):
    if cond.ndim == 1:
        cond = cond.view(1, -1)
    batch = cond.shape[0]
    if init_mode == "zero":
        x = torch.zeros((batch, coeff_dim), device=cond.device, dtype=cond.dtype)
    else:
        x = torch.randn((batch, coeff_dim), device=cond.device, dtype=cond.dtype)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        tval = (k + 0.5) / n_steps
        t = torch.full((batch, 1), tval, device=cond.device, dtype=cond.dtype)
        v = model(x, t, cond)
        x = x + dt * v
    return x


def path_and_lipschitz_regularization(model, cond_batch, coeff_dim, pca_components_t, pca_mean_t):
    n = cond_batch.shape[0]
    if n < 4:
        zero = cond_batch.sum() * 0.0
        return zero, zero

    m = min(PATH_REG_BATCH, n // 2)
    perm = torch.randperm(n, device=cond_batch.device)
    idx_a = perm[:m]
    idx_b = perm[-m:]

    z_a = cond_batch[idx_a]
    z_b = cond_batch[idx_b]
    alpha = torch.rand((m, 1), device=cond_batch.device, dtype=cond_batch.dtype)
    z_mid = (1.0 - alpha) * z_a + alpha * z_b

    coeff_a = cfm_generate_batch_differentiable(model, z_a, coeff_dim=coeff_dim, n_steps=PATH_EULER_STEPS)
    coeff_b = cfm_generate_batch_differentiable(model, z_b, coeff_dim=coeff_dim, n_steps=PATH_EULER_STEPS)
    coeff_mid = cfm_generate_batch_differentiable(model, z_mid, coeff_dim=coeff_dim, n_steps=PATH_EULER_STEPS)

    target_coeff_mid = ((1.0 - alpha) * coeff_a + alpha * coeff_b).detach()
    curve_mid = coeffs_to_curve_torch(coeff_mid, pca_components_t, pca_mean_t)
    target_curve_mid = coeffs_to_curve_torch(target_coeff_mid, pca_components_t, pca_mean_t)

    d_curve_mid = curve_mid[:, 1:] - curve_mid[:, :-1]
    d_target_mid = target_curve_mid[:, 1:] - target_curve_mid[:, :-1]

    path_loss = (
        F.mse_loss(coeff_mid, target_coeff_mid) +
        F.mse_loss(curve_mid, target_curve_mid) +
        PATH_DERIV_LAMBDA * F.mse_loss(d_curve_mid, d_target_mid)
    )

    delta = LIPSCHITZ_DELTA_STD * torch.randn_like(z_a)
    z_near = z_a + delta
    coeff_near = cfm_generate_batch_differentiable(model, z_near, coeff_dim=coeff_dim, n_steps=PATH_EULER_STEPS)
    curve_a = coeffs_to_curve_torch(coeff_a, pca_components_t, pca_mean_t)
    curve_near = coeffs_to_curve_torch(coeff_near, pca_components_t, pca_mean_t)

    slope = torch.norm((curve_near - curve_a).reshape(m, -1), dim=1) / (torch.norm(delta, dim=1) + 1e-8)
    lipschitz_loss = torch.mean(F.relu(slope - LIPSCHITZ_TARGET) ** 2)
    return path_loss, lipschitz_loss


def train_cfm(latent_full_train, coeffs_train, pca):
    x = torch.from_numpy(coeffs_train.astype(np.float32))
    c = torch.from_numpy(latent_full_train.astype(np.float32))
    ds = TensorDataset(x, c)
    dl = DataLoader(ds, batch_size=CFM_BATCH_SIZE, shuffle=True, drop_last=False)

    pca_components_t = torch.tensor(pca.components_.astype(np.float32), device=DEVICE)
    pca_mean_t = torch.tensor(pca.mean_.astype(np.float32), device=DEVICE)

    model = CFMTransformer(
        coeff_dim=coeffs_train.shape[1],
        cond_dim=latent_full_train.shape[1],
        hidden=CFM_HIDDEN,
        num_layers=CFM_LAYERS,
        dropout=CFM_DROPOUT,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFM_LR, weight_decay=CFM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=CFM_LR_PATIENCE, min_lr=1e-5
    )
    mse = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    epochs_no_improve = 0

    for ep in range(CFM_EPOCHS):
        model.train()
        losses = []
        for x1, cond in dl:
            x1 = x1.to(DEVICE)
            cond = cond.to(DEVICE)

            cond_aug = cond + CFM_COORD_NOISE_STD * torch.randn_like(cond)
            x1_aug = x1 + CFM_COEFF_NOISE_STD * torch.randn_like(x1)

            x0 = torch.randn_like(x1_aug)
            t = torch.rand((len(x1_aug), 1), device=DEVICE)
            xt = (1 - t) * x0 + t * x1_aug
            v_target = x1_aug - x0
            v_pred = model(xt, t, cond_aug)

            cond_near = cond_aug + CFM_LOCAL_DELTA_STD * torch.randn_like(cond_aug)
            v_near = model(xt, t, cond_near)
            smooth_loss = mse(v_pred, v_near)

            if PATH_REG_LAMBDA > 0.0 or LIPSCHITZ_LAMBDA > 0.0:
                path_loss, lipschitz_loss = path_and_lipschitz_regularization(
                    model, cond_aug, coeff_dim=coeffs_train.shape[1],
                    pca_components_t=pca_components_t, pca_mean_t=pca_mean_t
                )
            else:
                zero = v_pred.sum() * 0.0
                path_loss, lipschitz_loss = zero, zero

            loss = (
                mse(v_pred, v_target) +
                CFM_SMOOTH_LAMBDA * smooth_loss +
                PATH_REG_LAMBDA * path_loss +
                LIPSCHITZ_LAMBDA * lipschitz_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses))
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (ep + 1) % 50 == 0:
            lr = opt.param_groups[0]["lr"]
            print(f"CFM epoch {ep+1}/{CFM_EPOCHS} loss={epoch_loss:.5f} lr={lr:.2e}")

        if epochs_no_improve >= CFM_EARLY_STOP_PATIENCE:
            print(f"CFM early stopping at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def cfm_generate(model, cond_vec, coeff_dim, n_steps=N_EULER_STEPS, init_mode=None):
    model.eval()
    cond = torch.tensor(cond_vec, dtype=torch.float32, device=DEVICE).view(1, -1)
    mode = CFM_INIT_MODE if init_mode is None else init_mode
    if mode == "zero":
        x = torch.zeros((1, coeff_dim), device=DEVICE)
    else:
        x = torch.randn((1, coeff_dim), device=DEVICE)
    dt = 1.0 / n_steps
    with torch.no_grad():
        for k in range(n_steps):
            tval = (k + 0.5) / n_steps
            t = torch.full((1, 1), tval, device=DEVICE)
            v = model(x, t, cond)
            x = x + dt * v
    return x[0].cpu().numpy()


def reconstruct_curve_from_latent(cond_vec, cfm_model, pca, init_mode=None, n_steps=N_EULER_STEPS,
                                  latent_full_train=None, coeffs_train=None, anchor_blend=0.0):
    coeff_hat = cfm_generate(cfm_model, cond_vec, coeff_dim=pca.n_components_, n_steps=n_steps, init_mode=init_mode)
    if anchor_blend > 0.0 and latent_full_train is not None and coeffs_train is not None:
        coeff_anchor = infer_coeff_anchor_from_full_latent(cond_vec, latent_full_train, coeffs_train)
        coeff_hat = (1.0 - anchor_blend) * coeff_hat + anchor_blend * coeff_anchor
    recon = np.clip(pca.inverse_transform(coeff_hat[None, :])[0], 0.0, 1.0)
    return recon


def infer_full_latent_from_2d(query_xy, latent_xy_train, latent_full_train, center_xy=None, center_full=None):
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(1, 2)
    d2 = np.sum((latent_xy_train - query_xy) ** 2, axis=1)
    k = min(QUERY_NEIGHBORS, len(latent_xy_train))
    idx = np.argsort(d2)[:k]
    dsel = d2[idx]
    logits = -dsel / max(QUERY_LATENT_TAU ** 2, 1e-8)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    z_local = w @ latent_full_train[idx]

    if center_xy is not None and center_full is not None:
        centers2 = np.vstack([center_xy[b] for b in BRIDGE_NAMES])
        centersf = np.vstack([center_full[b] for b in BRIDGE_NAMES])
        w_center = softmax_weights(query_xy.ravel(), centers2, tau=BACKGROUND_TAU)
        z_center = w_center @ centersf
        z_full = (1.0 - QUERY_CENTER_BLEND) * z_local + QUERY_CENTER_BLEND * z_center
    else:
        z_full = z_local
    return z_full.astype(np.float32)


def infer_coeff_anchor_from_full_latent(query_z, latent_full_train, coeffs_train):
    query_z = np.asarray(query_z, dtype=np.float32).reshape(1, -1)
    d2 = np.sum((latent_full_train - query_z) ** 2, axis=1)
    k = min(COEFF_ANCHOR_K, len(latent_full_train))
    idx = np.argsort(d2)[:k]
    dsel = d2[idx]
    logits = -dsel / max(COEFF_ANCHOR_TAU ** 2, 1e-8)
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return (w @ coeffs_train[idx]).astype(np.float32)


# ============================================================
# EVALUATION HELPERS
# ============================================================
def encode_latents(model, features):
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(features.astype(np.float32)).to(DEVICE))
        coords2 = model.project(mu)
        return mu.cpu().numpy().astype(np.float32), coords2.cpu().numpy().astype(np.float32)


def compute_bridge_centers(latent_array, label_names):
    centers = {}
    for b in BRIDGE_NAMES:
        centers[b] = latent_array[label_names == b].mean(axis=0)
    return centers


def evaluate_center_reconstructions(data_dict, cfm_model, pca, train_centers_full):
    rows = []
    recon_at_centers = {}
    for b in BRIDGE_NAMES:
        recon = reconstruct_curve_from_latent(train_centers_full[b], cfm_model, pca)
        recon_at_centers[b] = recon
        true = data_dict[b].test_ref_curve
        rows.append({
            "bridge": b,
            "Pearson_r": pearson_corr(true, recon),
            "RMSE": rmse(true, recon),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "center_reconstruction_metrics_test.csv"), index=False)
    return df, recon_at_centers













# ============================================================
# INTERACTIVE PLOT
# ============================================================
def make_background(latent_xy_train, centers):
    x_min, x_max = latent_xy_train[:, 0].min(), latent_xy_train[:, 0].max()
    y_min, y_max = latent_xy_train[:, 1].min(), latent_xy_train[:, 1].max()
    padx = 0.20 * max(1e-6, x_max - x_min)
    pady = 0.20 * max(1e-6, y_max - y_min)
    x_min, x_max = x_min - padx, x_max + padx
    y_min, y_max = y_min - pady, y_max + pady

    xx = np.linspace(x_min, x_max, GRID_RES)
    yy = np.linspace(y_min, y_max, GRID_RES)
    bg = np.zeros((len(yy), len(xx), 3), dtype=float)
    center_mat = np.vstack([centers[b] for b in BRIDGE_NAMES])

    for iy, y in enumerate(yy):
        for ix, x in enumerate(xx):
            w = softmax_weights([x, y], center_mat, tau=BACKGROUND_TAU)
            w = sharpen_influence_weights(w, power=BACKGROUND_WEIGHT_POWER)
            rgb = blend_color(w, BRIDGE_NAMES)
            bg[iy, ix, :] = make_vivid_color(rgb, floor=BACKGROUND_COLOR_FLOOR)

    return bg, (x_min, x_max, y_min, y_max)


def build_custom_legend():
    handles = []
    for b in BRIDGE_NAMES:
        color = get_bridge_color(b)
        handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                              markerfacecolor=color, markeredgecolor=color, alpha=0.7,
                              label=f"{display_name(b)} train"))
        handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=8,
                              markerfacecolor='none', markeredgecolor=color, markeredgewidth=1.8,
                              label=f"{display_name(b)} test"))
    handles.append(Line2D([0], [0], marker='X', linestyle='None', markersize=11,
                          markerfacecolor='black', markeredgecolor='black',
                          label='Bridge center'))
    handles.append(Line2D([0], [0], marker='o', linestyle='None', markersize=9,
                          markerfacecolor='black', markeredgecolor='black',
                          label='Current query'))
    return handles


def add_scroll_zoom(fig, ax):
    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        scale_factor = 1 / ZOOM_BASE if event.button == 'up' else ZOOM_BASE

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)


def interactive_plot(common_freq, data_dict, latent_xy_train, latent_full_train, train_names,
                     latent_xy_test, latent_full_test, test_names,
                     train_centers_2d, train_centers_full, cfm_model, pca, coeffs_train):
    fig = plt.figure(figsize=(19, 10.5))
    gs = GridSpec(2, 2, width_ratios=[1.10, 1.25], height_ratios=[1.0, 0.55], figure=fig)
    ax_lat = fig.add_subplot(gs[:, 0])
    ax_sv1 = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    bg, extent = make_background(latent_xy_train, train_centers_2d)
    x_min, x_max, y_min, y_max = extent
    ax_lat.imshow(bg, extent=[x_min, x_max, y_min, y_max], origin='lower', alpha=BACKGROUND_ALPHA, aspect='auto')

    xx, yy, dominant_idx, dominant_strength = compute_influence_fields((x_min, x_max, y_min, y_max), train_centers_2d)
    for k in range(len(BRIDGE_NAMES)):
        mask = (dominant_idx == k).astype(float)
        ax_lat.contour(
            xx, yy, mask,
            levels=[BACKGROUND_CONTOUR_LEVEL],
            colors=BACKGROUND_CONTOUR_COLOR,
            linewidths=BACKGROUND_CONTOUR_WIDTH,
            alpha=BACKGROUND_CONTOUR_ALPHA,
        )

    add_background_explanation(ax_lat)

    for b in BRIDGE_NAMES:
        mask_tr = (train_names == b)
        mask_te = (test_names == b)
        ax_lat.scatter(latent_xy_train[mask_tr, 0], latent_xy_train[mask_tr, 1],
                       s=36, alpha=POINT_ALPHA_TRAIN, color=get_bridge_color(b), label=None)
        ax_lat.scatter(latent_xy_test[mask_te, 0], latent_xy_test[mask_te, 1],
                       s=50, alpha=POINT_ALPHA_TEST, facecolors='none', edgecolors=get_bridge_color(b),
                       linewidths=1.2, label=None)
        cx, cy = train_centers_2d[b]
        ax_lat.scatter([cx], [cy], s=270, color=get_bridge_color(b), edgecolor='k', linewidth=1.2, marker='X', zorder=10)

    ax_lat.set_title(
        "VAE latent space"
    )
    ax_lat.set_xlabel("Latent x")
    ax_lat.set_ylabel("Latent y")
    ax_lat.grid(alpha=0.20)
    ax_lat.set_xlim(x_min, x_max)
    ax_lat.set_ylim(y_min, y_max)
    ax_lat.legend(handles=build_custom_legend(), loc='upper right', fontsize=14, ncol=1, framealpha=0.60)
    add_scroll_zoom(fig, ax_lat)

    query_pt, = ax_lat.plot([], [], marker='o', color='black', ms=9, zorder=12)
    center_mat = np.vstack([train_centers_2d[b] for b in BRIDGE_NAMES])
    test_curves_all = np.vstack([data_dict[b].test_curves for b in BRIDGE_NAMES]).astype(np.float32)

    def draw_info_panel(coord, nearest_bridge, nearest_test_bridge, w, pearson_ref, rmse_ref, pearson_test, rmse_test):
        ax_info.clear()
        order = np.argsort(-w)
        ordered_bridges = [BRIDGE_NAMES[i] for i in order]
        ordered_weights = w[order]
        ordered_colors = [get_bridge_color(b) for b in ordered_bridges]

        y_pos = np.arange(len(ordered_bridges))
        bars = ax_info.barh(y_pos, ordered_weights, color=ordered_colors, alpha=0.92)
        ax_info.set_yticks(y_pos)
        ax_info.set_yticklabels([display_name(b) for b in ordered_bridges], fontsize=16)
        ax_info.invert_yaxis()
        ax_info.set_xlim(0, 1.06)
        ax_info.set_xlabel("Cluster influence weight")
        ax_info.grid(axis='x', alpha=0.25)
        ax_info.set_title("Local bridge influence around the queried point", fontsize=19)

        for bar, val in zip(bars, ordered_weights):
            x = float(bar.get_width())
            y = float(bar.get_y() + bar.get_height() / 2.0)
            label = f"{val:.3f}"
            if x < 0.88:
                ax_info.text(x + 0.015, y, label, va='center', ha='left', fontsize=15, weight='bold')
            else:
                ax_info.text(x - 0.015, y, label, va='center', ha='right', fontsize=15, weight='bold', color='white')

    def update_query(qx, qy):
        coord = np.array([qx, qy], dtype=np.float32)
        full_latent = infer_full_latent_from_2d(
            coord, latent_xy_train, latent_full_train,
            center_xy=train_centers_2d, center_full=train_centers_full
        )
        recon = reconstruct_curve_from_latent(
            full_latent, cfm_model, pca,
            latent_full_train=latent_full_train,
            coeffs_train=coeffs_train,
            anchor_blend=COEFF_ANCHOR_BLEND,
        )

        d2_test = np.sum((latent_xy_test - coord[None, :]) ** 2, axis=1)
        idx_test = int(np.argmin(d2_test))
        nearest_test_curve = test_curves_all[idx_test]
        nearest_test_bridge = str(test_names[idx_test])

        d2_center = np.sum((center_mat - coord[None, :]) ** 2, axis=1)
        idx_center = int(np.argmin(d2_center))
        nearest_bridge = BRIDGE_NAMES[idx_center]
        nearest_center_ref = data_dict[nearest_bridge].test_ref_curve

        w = softmax_weights(coord, center_mat, tau=BACKGROUND_TAU)
        w = sharpen_influence_weights(w, power=BACKGROUND_WEIGHT_POWER)

        pearson_test = pearson_corr(nearest_test_curve, recon)
        rmse_test = rmse(nearest_test_curve, recon)
        pearson_ref = pearson_corr(nearest_center_ref, recon)
        rmse_ref = rmse(nearest_center_ref, recon)

        ax_sv1.clear()
        ax_sv1.plot(common_freq, recon, color='black', lw=3.0, label='Queried CFM reconstruction')
        ax_sv1.plot(common_freq, nearest_center_ref, color=get_bridge_color(nearest_bridge), lw=2.7,
                    label=f"Nearest bridge reference spectrum: {display_name(nearest_bridge)}")
        ax_sv1.plot(common_freq, nearest_test_curve, color='0.45', lw=2.1, ls='--',
                    label=f"Nearest held-out test window: {display_name(nearest_test_bridge)}")
        ax_sv1.scatter([common_freq[np.argmax(recon)]], [float(np.max(recon))], c='k', s=40, zorder=5)

        ax_sv1.set_title(
            f"Query ({qx:.3f}, {qy:.3f})  |  Metric: Pearson correlation\n"
            f" queried vs nearest full bridge = {pearson_ref:.3f} | queried vs nearest test window = {pearson_test:.3f}",
            fontsize=16,
        )
        ax_sv1.set_xlabel("Frequency (Hz)")
        ax_sv1.set_ylabel("Normmalized amplitude")
        ax_sv1.set_xlim(common_freq.min(), common_freq.max())
        ax_sv1.set_ylim(0.0, 1.02)
        ax_sv1.grid(alpha=0.25)
        ax_sv1.legend(fontsize=16, ncol=1, loc='lower left', framealpha=0.80)

        draw_info_panel(coord, nearest_bridge, nearest_test_bridge, w, pearson_ref, rmse_ref, pearson_test, rmse_test)
        fig.canvas.draw_idle()

    first = train_centers_2d[BRIDGE_NAMES[0]]
    query_pt.set_data([first[0]], [first[1]])
    update_query(first[0], first[1])

    state = {"dragging": False, "last_update": 0.0}

    def handle_update(event):
        if event.inaxes != ax_lat or event.xdata is None or event.ydata is None:
            return
        now = time.time()
        if now - state["last_update"] < DRAG_THROTTLE_SEC:
            return
        state["last_update"] = now
        qx, qy = float(event.xdata), float(event.ydata)
        query_pt.set_data([qx], [qy])
        update_query(qx, qy)

    def on_press(event):
        if event.button == 1 and event.inaxes == ax_lat:
            state["dragging"] = True
            handle_update(event)

    def on_motion(event):
        if state["dragging"]:
            handle_update(event)

    def on_release(event):
        if event.button == 1:
            state["dragging"] = False

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.tight_layout()
    preview_path = os.path.join(OUTPUT_DIR, "interactive_preview_robust.png")
    plt.savefig(preview_path, dpi=180)
    print(f"Saved preview to: {preview_path}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    common_freq, data_dict = load_all_bridges()

    (
        train_curves, train_features, train_labels_idx, train_names,
        test_curves, test_features, test_labels_idx, test_names,
    ) = build_split_datasets(data_dict)

    print("\nTraining robust VAE with improved latent geometry...")
    vae_model = train_vae(train_curves, train_features, train_labels_idx)

    latent_full_train, latent_xy_train = encode_latents(vae_model, train_features)
    latent_full_test, latent_xy_test = encode_latents(vae_model, test_features)

    train_centers_2d = compute_bridge_centers(latent_xy_train, train_names)
    train_centers_full = compute_bridge_centers(latent_full_train, train_names)

    pd.DataFrame({
        "bridge": BRIDGE_NAMES,
        "x": [train_centers_2d[b][0] for b in BRIDGE_NAMES],
        "y": [train_centers_2d[b][1] for b in BRIDGE_NAMES],
    }).to_csv(os.path.join(OUTPUT_DIR, "bridge_train_latent_centers_2d.csv"), index=False)

    pd.DataFrame({
        "bridge": BRIDGE_NAMES,
        **{f"z{i+1}": [train_centers_full[b][i] for b in BRIDGE_NAMES] for i in range(latent_full_train.shape[1])},
    }).to_csv(os.path.join(OUTPUT_DIR, "bridge_train_latent_centers_full.csv"), index=False)

    pca_dim = min(PCA_DIM, train_curves.shape[0] - 1, train_curves.shape[1])
    pca_dim = max(8, pca_dim)
    pca = PCA(n_components=pca_dim, random_state=SEED)
    coeffs_train = pca.fit_transform(train_curves).astype(np.float32)
    pd.DataFrame({"explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
        os.path.join(OUTPUT_DIR, "pca_explained_variance.csv"), index=False
    )

    print("\nTraining smooth conditional flow matching transformer on full latent...")
    cfm_model = train_cfm(latent_full_train, coeffs_train, pca)

    print("\nKnown-center reconstruction summary...")
    center_metrics, _ = evaluate_center_reconstructions(data_dict, cfm_model, pca, train_centers_full)
    print(center_metrics.to_string(index=False))

    print("\nOpening interactive latent map...")
    interactive_plot(
        common_freq, data_dict,
        latent_xy_train, latent_full_train, train_names,
        latent_xy_test, latent_full_test, test_names,
        train_centers_2d, train_centers_full, cfm_model, pca, coeffs_train
    )
