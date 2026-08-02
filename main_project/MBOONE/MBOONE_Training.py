"""Unsupervised convolutional autoencoder for MicroBooNE collection-plane image
tiles (arXiv:2509.21817, Fig. 3a/3b).

Same architecture and reconstruction-loss training procedure as
tutorial/MBOONE/training.py, with two deliberate changes for this project:

  1. NOT called "teacher". We do no knowledge distillation here -- there is no
     Student, no qkeras, no quantization. It is simply an unsupervised
     autoencoder trained to reconstruct its input; the reconstruction MSE is the
     anomaly score. (The layer names use an "unsup_" prefix; the parameter counts
     are unchanged and still match paper Table 1.)

  2. Trains on the NON-EMPTY tile files written by remove_empty.py (default the
     charge >= 50 set in nonEmptyImages2/), streamed straight from HDF5 -- NOT
     from dense .npy. Converting these tiles to dense .npy would blow 5.3 GB up
     to ~227 GB (186 GB for 864x64 alone) because a kept tile is still ~99%
     zeros; gzip-chunked HDF5 keeps it at 5 GB with lazy per-tile access. The
     tiles are already non-empty by construction, so training.py's --drop-empty
     is unnecessary here.

Loading: a keras.utils.Sequence reads tiles on demand from the 18 per-file .h5
datasets, so RAM stays flat no matter how many files you train on. The 50/10/40
train/val/test split is done ONCE globally over all tiles (like training.py's
train_test_split, but across all files at once) with a fixed seed.

By default it trains all three sizes (864x64, 64x32, 18x16) in one run.
Outputs go to a per-threshold subfolder of savedModel/ (--output-dir), so the
charge>0 and charge>=50 runs never overwrite. The subfolder name is the charge
threshold read from the tile files themselves:
    savedModel/Q50/unsupervised_864x64.h5           (best model, min val_loss)
    savedModel/Q50/unsupervised_864x64_history.csv  (epoch,loss,val_loss -- like the tutorial)
"Q50" = nonEmptyImages2 (charge>=50); "Q0" = nonEmptyImages (charge>0).
Everything is CPU-only (GPUs hidden before TF import).

Usage:
  python MBOONE_Training.py                         # all sizes, nonEmptyImages2, 20 epochs
  python MBOONE_Training.py --sizes 864x64 --epochs 20
  python MBOONE_Training.py --data-dir nonEmptyImages   # the charge > 0 set instead
  python MBOONE_Training.py --limit 20000 --epochs 2    # fast smoke test
  python MBOONE_Training.py --self-test                 # architecture checks, no data
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU only: hide GPUs before TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import h5py

from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Conv2D, Dense, Flatten, Input, Reshape, UpSampling2D,
)
from tensorflow.keras.models import Model

# --------------------------------- Configuration ---------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "nonEmptyImages2"        # charge >= 50 set
ALL_SIZES = [(864, 64), (64, 32), (18, 16)]

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.5, 0.1, 0.4

# Same layer inventory as training.py, "teacher_" -> "unsup_". Counts unchanged.
EXPECTED_LAYER_NAMES = [
    "unsup_inputs_", "unsup_reshape", "unsup_conv2d_1", "unsup_relu_1",
    "unsup_pool_1", "unsup_conv2d_2", "unsup_relu_2", "unsup_flatten",
    "unsup_latent", "unsup_dense", "unsup_reshape2", "unsup_relu_3",
    "unsup_conv2d_3", "unsup_relu_4", "unsup_upsampling", "unsup_conv2d_4",
    "unsup_relu_5", "unsup_outputs",
]
KNOWN_PARAM_COUNTS = {(18, 16, 1): 367201, (64, 32, 1): 2492401, (864, 64, 1): 66789361}
# ---------------------------------------------------------------------------------


def build_unsupervised(segment_shape: Tuple[int, int, int]) -> keras.Model:
    """The autoencoder from training.py / paper Fig. 3a-b, parametrized by tile shape.

    Encoder Conv(20)->relu->avgpool->Conv(30)->relu->flatten->Dense(80) bottleneck;
    decoder Dense->reshape->relu->Conv(30)->relu->upsample->Conv(20)->relu->Conv(1,relu).
    Both spatial dims must be even so pool((2,2)) then upsample((2,2)) round-trips.
    """
    W, H, C = segment_shape
    if W % 2 or H % 2:
        raise ValueError("segment spatial dims must be even; got %r" % (segment_shape,))
    w2, h2 = W // 2, H // 2

    inputs = Input(shape=segment_shape, name="unsup_inputs_")
    x = Reshape((W, H, C), name="unsup_reshape")(inputs)
    x = Conv2D(20, (3, 3), strides=1, padding="same", name="unsup_conv2d_1")(x)
    x = Activation("relu", name="unsup_relu_1")(x)
    x = AveragePooling2D((2, 2), name="unsup_pool_1")(x)
    x = Conv2D(30, (3, 3), strides=1, padding="same", name="unsup_conv2d_2")(x)
    x = Activation("relu", name="unsup_relu_2")(x)
    x = Flatten(name="unsup_flatten")(x)
    x = Dense(80, activation="relu", name="unsup_latent")(x)
    x = Dense(w2 * h2 * 30, name="unsup_dense")(x)
    x = Reshape((w2, h2, 30), name="unsup_reshape2")(x)
    x = Activation("relu", name="unsup_relu_3")(x)
    x = Conv2D(30, (3, 3), strides=1, padding="same", name="unsup_conv2d_3")(x)
    x = Activation("relu", name="unsup_relu_4")(x)
    x = UpSampling2D((2, 2), name="unsup_upsampling")(x)
    x = Conv2D(20, (3, 3), strides=1, padding="same", name="unsup_conv2d_4")(x)
    x = Activation("relu", name="unsup_relu_5")(x)
    outputs = Conv2D(1, (3, 3), activation="relu", strides=1, padding="same",
                     name="unsup_outputs")(x)
    return Model(inputs, outputs, name="Unsupervised")


def total_pixel_intensity(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=(1, 2, 3))


def anomaly_score(model: keras.Model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Per-tile reconstruction MSE (higher = more anomalous)."""
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean((x - recon) ** 2, axis=(1, 2, 3))


def scaled_anomaly_score(model: keras.Model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """anomaly_score / total pixel intensity (paper 3.1.5); empty tiles -> 0."""
    score = anomaly_score(model, x, batch_size=batch_size)
    total = total_pixel_intensity(x)
    out = np.zeros_like(score)
    nz = total > 0
    out[nz] = score[nz] / total[nz]
    return out


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("[epoch %d] %.1fs  loss=%.6g  val_loss=%.6g"
              % (epoch + 1, time.time() - self._t0,
                 logs.get("loss", float("nan")), logs.get("val_loss", float("nan"))))


class TileSequence(keras.utils.Sequence):
    """Streams (X, X) autoencoder batches of tiles from the per-file HDF5 datasets.

    A batch may draw tiles from several files; reads are grouped per file and
    issued in sorted order (h5py fancy-index requirement), then scattered back
    into batch order. Never holds more than one batch of pixel data in memory.
    """

    def __init__(self, tiles_dsets: List[h5py.Dataset], file_idx: np.ndarray,
                 row_idx: np.ndarray, order: np.ndarray, batch_size: int,
                 seg_shape: Tuple[int, int, int], shuffle: bool, seed: int):
        super().__init__()          # Keras 3 PyDataset base (workers/queue kwargs)
        self.tiles = tiles_dsets
        self.file_idx = file_idx
        self.row_idx = row_idx
        self.order = np.asarray(order).copy()
        self.bs = batch_size
        self.sw, self.st, _ = seg_shape
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.order) / self.bs))

    def _load(self, sel: np.ndarray) -> np.ndarray:
        f = self.file_idx[sel]
        r = self.row_idx[sel]
        X = np.empty((len(sel), self.sw, self.st, 1), dtype=np.float32)
        for fi in np.unique(f):
            m = f == fi
            rows = r[m]
            srt = np.argsort(rows)
            data = self.tiles[fi][np.sort(rows)]           # (k, sw, st), sorted read
            X[np.nonzero(m)[0][srt], :, :, 0] = data        # scatter back to batch order
        return X

    def __getitem__(self, b):
        sel = self.order[b * self.bs:(b + 1) * self.bs]
        X = self._load(sel)
        return X, X

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def open_size(data_dir: Path, sw: int, st: int, files) -> Tuple[list, np.ndarray, np.ndarray, str]:
    """Open the per-file tile datasets for one size and build the global tile table.

    Returns (open h5py.File handles, file_idx per tile, row_idx per tile, keep_rule).
    """
    handles, dsets, fidx, ridx = [], [], [], []
    keep_rule = "?"
    charge_min = 0.0
    for k, fnum in enumerate(files):
        p = data_dir / ("bnb_WithWire_%02d_plane2_%dx%d_nonempty.h5" % (fnum, sw, st))
        if not p.exists():
            raise SystemExit("missing tile file: %s (run remove_empty.py first)" % p)
        h = h5py.File(str(p), "r")
        handles.append(h)
        dsets.append(h["tiles"])
        n = h["tiles"].shape[0]
        fidx.append(np.full(n, k, dtype=np.int32))
        ridx.append(np.arange(n, dtype=np.int64))
        keep_rule = h.attrs.get("keep_rule", keep_rule)
        charge_min = float(h.attrs.get("charge_min", charge_min))   # 0 for the charge>0 set
    return handles, dsets, np.concatenate(fidx), np.concatenate(ridx), keep_rule, charge_min


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle 0..n-1 and cut a 50/10/40 train/val/test split (fixed by seed)."""
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_tr = int(round(TRAIN_RATIO * n))
    n_va = int(round(VAL_RATIO * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def run_self_test() -> None:
    for shape in [(18, 16, 1), (64, 32, 1), (864, 64, 1)]:
        m = build_unsupervised(shape)
        names = [l.name for l in m.layers]
        assert names == EXPECTED_LAYER_NAMES, "layer names differ: %r" % names
        out = m.predict(np.zeros((1,) + shape, dtype="float32"), verbose=0)
        assert out.shape == (1,) + shape, "shape round-trip failed: %r" % (out.shape,)
        n = m.count_params()
        assert n == KNOWN_PARAM_COUNTS[shape], "param count %d != %d" % (n, KNOWN_PARAM_COUNTS[shape])
        r = m.predict((np.random.rand(4, *shape) * 100).astype("float32"), verbose=0)
        assert (r >= 0).all(), "reconstruction has negative pixels"
        print("[ok] %r: 18 layers, %d params (paper Table 1), shape round-trips, output >= 0" % (shape, n))
    print("SELF-TEST PASSED")


def eval_anomaly_sample(model, seq: TileSequence, max_tiles: int = 50000) -> None:
    """Percentiles of the anomaly score over up to max_tiles held-out test tiles."""
    scores = []
    seen = 0
    for b in range(len(seq)):
        X, _ = seq[b]
        recon = model.predict(X, verbose=0)
        scores.append(np.mean((X - recon) ** 2, axis=(1, 2, 3)))
        seen += len(X)
        if seen >= max_tiles:
            break
    if not scores:
        return
    s = np.concatenate(scores)
    med, p90, p99, mx = np.percentile(s, [50, 90, 99, 100])
    print("anomaly score on %d held-out test tiles: median=%.4g  p90=%.4g  p99=%.4g  max=%.4g"
          % (len(s), med, p90, p99, mx))
    print("(these tiles are all non-empty, so unlike the paper's full set the median is not "
          "near 0; expect a right-skewed distribution with a long tail toward high-multiplicity "
          "tiles. A flat/broad spread usually means the pixel values got normalized somewhere.)")


def train_one_size(sw: int, st: int, args: argparse.Namespace) -> None:
    seg_shape = (sw, st, 1)
    files = list(range(args.files))
    print("\n" + "=" * 72)
    print("SIZE %dx%d  |  data=%s  |  files=%d" % (sw, st, args.data_dir.name, len(files)))

    handles, dsets, file_idx, row_idx, keep_rule, charge_min = open_size(args.data_dir, sw, st, files)
    # Threshold tag baked into the output names so the charge>0 and charge>=50
    # models/histories never overwrite each other. nonEmptyImages2 -> "Q50".
    tag = "Q%d" % int(round(charge_min))
    n = file_idx.size
    if args.limit is not None:
        n = min(n, args.limit)
    tr, va, te = split_indices(n, args.seed)
    print("keep_rule: %s  ->  tag %s  |  tiles: %d  (train=%d val=%d test=%d)"
          % (keep_rule, tag, n, len(tr), len(va), len(te)))

    model = build_unsupervised(seg_shape)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    print("parameters: %d" % model.count_params())

    out_dir = args.output_dir / tag           # e.g. savedModel/Q50/
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("unsupervised_%dx%d.h5" % (sw, st))
    csv_path = out_dir / ("unsupervised_%dx%d_history.csv" % (sw, st))
    if csv_path.exists():
        csv_path.unlink()

    train_seq = TileSequence(dsets, file_idx, row_idx, tr, args.batch_size, seg_shape,
                             shuffle=True, seed=args.seed)
    val_seq = TileSequence(dsets, file_idx, row_idx, va, args.batch_size, seg_shape,
                           shuffle=False, seed=args.seed)

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(str(csv_path), append=False),
        EpochTimer(),
    ]
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs,
              callbacks=callbacks, verbose=2)

    print("best model (min val_loss) -> %s" % out_path)
    print("loss history -> %s" % csv_path)

    if len(te):
        test_seq = TileSequence(dsets, file_idx, row_idx, te, args.batch_size, seg_shape,
                                shuffle=False, seed=args.seed)
        eval_anomaly_sample(model, test_seq)

    for h in handles:
        h.close()
    keras.backend.clear_session()
    gc.collect()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the unsupervised autoencoder on non-empty tiles (CPU, streamed from HDF5).")
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="folder of *_nonempty.h5 files (default: nonEmptyImages2 = charge>=50)")
    p.add_argument("--sizes", type=str, default="864x64,64x32,18x16",
                   help="comma-separated tile sizes to train (default all three)")
    p.add_argument("--files", type=int, default=18, help="number of source files 0..N-1 (default 18)")
    p.add_argument("--epochs", type=int, default=20, help="epochs per size (default 20)")
    p.add_argument("--batch-size", type=int, default=128, help="mini-batch size (default 128)")
    p.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "savedModel"),
                   help="where to save models (default: main_project/MBOONE/savedModel)")
    p.add_argument("--seed", type=int, default=42, help="split shuffle seed (default 42)")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N tiles per size (fast smoke test). Default: all")
    p.add_argument("--self-test", action="store_true",
                   help="architecture checks only (no data, no training) and exit")
    a = p.parse_args(argv)
    a.data_dir = Path(a.data_dir)
    a.output_dir = Path(a.output_dir)
    a.sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes.split(",")]
    return a


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    for sw, st in args.sizes:
        if (sw, st) not in ALL_SIZES:
            raise SystemExit("unsupported size %dx%d (supported: %s)"
                             % (sw, st, ", ".join("%dx%d" % s for s in ALL_SIZES)))
        train_one_size(sw, st, args)


if __name__ == "__main__":
    main()
