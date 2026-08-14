import os
import sys


CPU_THREADS = "6"     # Change accordingly for whatever CPU you have
                         #   per-run override:  --threads N   or   CLUSTERING_THREADS=N
LOW_PRIORITY = True      # run at below-normal OS priority so foreground work preempts it
                         #   (turn off with  --priority normal)


def _resolve_threads() -> int:
    v = os.environ.get("CLUSTERING_THREADS")
    if "--threads" in sys.argv:                       # peek argv before argparse exists
        try:
            v = sys.argv[sys.argv.index("--threads") + 1]
        except (IndexError, ValueError):
            pass
    if v is None:
        v = CPU_THREADS
    if str(v).strip().lower() in ("auto", "", "0"):
        return max(1, (os.cpu_count() or 2) // 2)
    return max(1, int(v))


_THREADS = _resolve_threads()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, str(_THREADS))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU only: hide GPUs before TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# ========================================================================

import argparse
import gc
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Conv2D, Dense, Flatten, Input, Reshape, UpSampling2D,
)
from tensorflow.keras.models import Model

# Belt-and-suspenders: also cap TF's own op threading (honored on TF>=1.15/2.x; the env
# vars above already cover MKL/OpenMP and older TF).
try:
    tf.config.threading.set_intra_op_parallelism_threads(_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(_THREADS)
except Exception:
    pass


def _set_low_priority() -> bool:
    """Best-effort: drop to below-normal OS priority so foreground processes preempt us."""
    try:
        if os.name == "nt":
            import ctypes
            below_normal = 0x00004000
            proc = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(ctypes.windll.kernel32.SetPriorityClass(proc, below_normal))
        os.nice(10)
        return True
    except Exception:
        return False

# ============================== CONFIG (edit these) ==============================
# Every constant here is also a CLI flag with the same default, so you can either edit
# the number below or pass e.g. --epochs 20 / --percentile 50 -- whichever is easier.
PERCENTILE = 75          # which padded set to train on: 50 (144x112) or 75 (288x176)
DATASET = "binary"         # which cluster set: "crop" (pass-2 clusters_padded_*, oversized
                         # cropped), "downscale" (pass-3 clusters_padded_downscaled_*,
                         # oversized DOWNSCALED + <=4-tile blips dropped), or "binary"
                         # (clusters_padded_binary_*, charge -> 0/100; binary_cluster.py,
                         # 75th only). Anything else aborts (also --dataset crop|downscale|binary).
TOTAL_IMAGES = 100000     # clusters used for the WHOLE train/val/test split
                         # (train = 50% of this -> 30,000 at the default)
EPOCHS = 24              # training epochs (bump to 20 with no other change)
BATCH_SIZE = 128         # mini-batch size
SEED = 42                # subset + split + shuffle seed (reproducible)
TILE_TAG = "18x16"       # clustering tile size the padded file was made with
# ================================================================================

# Which padded HDF5 each dataset reads, and the tag its models are saved under (so a
# "crop" run and a "downscale" run never overwrite each other).  "crop" keeps the
# original unprefixed names for backward compatibility with the models already trained.
DATASET_FILES = {"crop": "clusters_padded_%d_%s.h5",
                 "downscale": "clusters_padded_downscaled_%d_%s.h5",
                 "binary": "clusters_padded_binary_%d_%s.h5"}
DATASET_TAG = {"crop": "", "downscale": "downscaled_", "binary": "binary_"}
DATASET_MAKE = {"crop": "python tile_cluster.py pass2",
                "downscale": "python tile_cluster.py pass3",
                "binary": "python binary_cluster.py"}


def _check_dataset(ds):
    if ds not in DATASET_FILES:
        raise SystemExit("--dataset must be one of %s, got %r"
                         % (sorted(DATASET_FILES), ds))
    return ds

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.5, 0.1, 0.4  

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "clusterImages"          # holds clusters_padded_*.h5
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "savedModel"

# Reference padded sizes for --self-test only; the trainer reads the true shape from the
# file, so these never gate an actual run (they just let the no-data test build a model).
CLUSTER_SIZES = {50: (144, 112), 75: (288, 176)}

# Same layer inventory as MBOONE_Training.py ("unsup_"). Architecture is UNCHANGED.
EXPECTED_LAYER_NAMES = [
    "unsup_inputs_", "unsup_reshape", "unsup_conv2d_1", "unsup_relu_1",
    "unsup_pool_1", "unsup_conv2d_2", "unsup_relu_2", "unsup_flatten",
    "unsup_latent", "unsup_dense", "unsup_reshape2", "unsup_relu_3",
    "unsup_conv2d_3", "unsup_relu_4", "unsup_upsampling", "unsup_conv2d_4",
    "unsup_relu_5", "unsup_outputs",
]
# Param counts at the reference tile sizes (paper Table 1) -- used to PROVE the copied
# architecture is byte-for-byte the same one. The cluster sizes are not in this table.
KNOWN_PARAM_COUNTS = {(18, 16, 1): 367201, (64, 32, 1): 2492401, (864, 64, 1): 66789361}


def build_unsupervised(segment_shape: Tuple[int, int, int]) -> keras.Model:
    """The autoencoder from MBOONE_Training.py / paper Fig. 3a-b, parametrized by input shape.

    UNCHANGED from MBOONE_Training.build_unsupervised. Encoder Conv(20)->relu->avgpool->
    Conv(30)->relu->flatten->Dense(80) bottleneck; decoder Dense->reshape->relu->Conv(30)->
    relu->upsample->Conv(20)->relu->Conv(1,relu). Both spatial dims must be even so
    pool((2,2)) then upsample((2,2)) round-trips (144,112,288,176 all qualify).
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
    """Per-cluster reconstruction MSE (higher = more anomalous)."""
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean((x - recon) ** 2, axis=(1, 2, 3))


def scaled_anomaly_score(model: keras.Model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """anomaly_score / total pixel intensity (paper 3.1.5); empty images -> 0."""
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


class ClusterSequence(keras.utils.Sequence):
    """Streams (X, X) autoencoder batches from the single padded `images` dataset.

    Holds only row indices; each batch is read from HDF5 in sorted order (h5py fancy-index
    requirement) and scattered back into batch order. Never keeps more than one batch of
    pixels in memory.
    """

    def __init__(self, images_dset: h5py.Dataset, order: np.ndarray, batch_size: int,
                 seg_shape: Tuple[int, int, int], shuffle: bool, seed: int):
        super().__init__()
        self.images = images_dset
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
        srt = np.argsort(sel)                       # permutation that sorts the row ids
        data = self.images[np.sort(sel)]           # (k, sw, st), sorted h5py read
        X = np.empty((len(sel), self.sw, self.st, 1), dtype=np.float32)
        X[srt, :, :, 0] = data                      # scatter back to batch order
        return X

    def __getitem__(self, b):
        sel = self.order[b * self.bs:(b + 1) * self.bs]
        X = self._load(sel)
        return X, X

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def open_padded(data_dir: Path, pct: int, tile_tag: str, dataset: str = "crop"):
    """Open the combined padded file for one percentile; return (handle, images, N, sw, st).

    ``dataset`` picks which file: "crop" -> clusters_padded_<pct>_<tag>.h5 (pass 2),
    "downscale" -> clusters_padded_downscaled_<pct>_<tag>.h5 (pass 3).
    """
    _check_dataset(dataset)
    p = data_dir / (DATASET_FILES[dataset] % (pct, tile_tag))
    if not p.exists():
        raise SystemExit("missing padded file: %s\n(run:  %s)" % (p, DATASET_MAKE[dataset]))
    h = h5py.File(str(p), "r")
    imgs = h["images"]
    n, sw, st = imgs.shape
    return h, imgs, int(n), int(sw), int(st)


def select_and_split(n_total: int, total: int, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly pick ``total`` clusters from 0..n_total-1, then cut 50/10/40 train/val/test.

    Returns row indices into the `images` dataset (fixed by ``seed``).
    """
    idx = np.arange(n_total)
    np.random.default_rng(seed).shuffle(idx)
    idx = idx[:min(total, n_total)]                 # random working subset
    m = len(idx)
    n_tr = int(round(TRAIN_RATIO * m))
    n_va = int(round(VAL_RATIO * m))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def run_self_test() -> None:
    """Prove the copied architecture is unchanged, and that it builds at the cluster sizes."""
    # 1. reference tile sizes: exact layer names + paper param counts must still match.
    for shape in [(18, 16, 1), (64, 32, 1), (864, 64, 1)]:
        m = build_unsupervised(shape)
        assert [l.name for l in m.layers] == EXPECTED_LAYER_NAMES, "layer names differ"
        n = m.count_params()
        assert n == KNOWN_PARAM_COUNTS[shape], "param count %d != %d" % (n, KNOWN_PARAM_COUNTS[shape])
        print("[ok] %r: 18 layers, %d params (paper Table 1) -- architecture unchanged" % (shape, n))
        keras.backend.clear_session()

    # 2. cluster sizes: 18-layer graph builds, round-trips shape, output >= 0.
    for pct, (sw, st) in CLUSTER_SIZES.items():
        shape = (sw, st, 1)
        m = build_unsupervised(shape)
        out = m.predict(np.zeros((1,) + shape, dtype="float32"), verbose=0)
        assert out.shape == (1,) + shape, "shape round-trip failed: %r" % (out.shape,)
        r = m.predict((np.random.rand(4, *shape) * 100).astype("float32"), verbose=0)
        assert (r >= 0).all(), "reconstruction has negative pixels"
        print("[ok] %dth pct %dx%d: 18 layers, %d params, shape round-trips, output >= 0"
              % (pct, sw, st, m.count_params()))
        keras.backend.clear_session()
    print("SELF-TEST PASSED")


def eval_anomaly_sample(model, seq: ClusterSequence, max_images: int = 50000) -> None:
    """Percentiles of the anomaly score over up to max_images held-out test clusters."""
    scores, seen = [], 0
    for b in range(len(seq)):
        X, _ = seq[b]
        recon = model.predict(X, verbose=0)
        scores.append(np.mean((X - recon) ** 2, axis=(1, 2, 3)))
        seen += len(X)
        if seen >= max_images:
            break
    if not scores:
        return
    s = np.concatenate(scores)
    med, p90, p99, mx = np.percentile(s, [50, 90, 99, 100])
    print("anomaly score on %d held-out test clusters: median=%.4g  p90=%.4g  p99=%.4g  max=%.4g"
          % (len(s), med, p90, p99, mx))


def train_one_percentile(pct: int, args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    h, imgs, n_total, sw, st = open_padded(args.data_dir, pct, args.tile_tag, args.dataset)
    seg_shape = (sw, st, 1)
    tr, va, te = select_and_split(n_total, args.total, args.seed)
    print("PERCENTILE %d  |  dataset=%s  |  images %dx%d  |  available=%d  |  using=%d  (train=%d val=%d test=%d)"
          % (pct, args.dataset, sw, st, n_total, len(tr) + len(va) + len(te), len(tr), len(va), len(te)))

    model = build_unsupervised(seg_shape)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    print("parameters: %d  |  epochs: %d  |  batch: %d" % (model.count_params(), args.epochs, args.batch_size))

    vtag = DATASET_TAG[args.dataset]                                       # "" (crop) or "downscaled_"
    out_dir = args.output_dir / ("cluster_%s%d_%s" % (vtag, pct, args.tile_tag))  # savedModel/cluster_[downscaled_]75_18x16/
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("unsupervised_cluster_%s%d_%s.h5" % (vtag, pct, args.tile_tag))
    csv_path = out_dir / ("unsupervised_cluster_%s%d_%s_history.csv" % (vtag, pct, args.tile_tag))
    if csv_path.exists():
        csv_path.unlink()

    train_seq = ClusterSequence(imgs, tr, args.batch_size, seg_shape, shuffle=True, seed=args.seed)
    val_seq = ClusterSequence(imgs, va, args.batch_size, seg_shape, shuffle=False, seed=args.seed)

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(str(csv_path), append=False),
        EpochTimer(),
    ]
    model.fit(train_seq, validation_data=val_seq, epochs=args.epochs, callbacks=callbacks, verbose=2)

    print("best model (min val_loss) -> %s" % out_path)
    print("loss history            -> %s" % csv_path)

    if len(te):
        test_seq = ClusterSequence(imgs, te, args.batch_size, seg_shape, shuffle=False, seed=args.seed)
        eval_anomaly_sample(model, test_seq)

    h.close()
    keras.backend.clear_session()
    gc.collect()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the unsupervised autoencoder on padded cluster images (CPU, streamed from HDF5).")
    p.add_argument("--percentile", type=str, default=str(PERCENTILE),
                   help="which padded set(s) to train: 50, 75, or '50,75' for both (default %d)" % PERCENTILE)
    p.add_argument("--dataset", type=str, default=DATASET, choices=sorted(DATASET_FILES),
                   help="'crop' (pass-2 clusters_padded_*), 'downscale' (pass-3 "
                        "clusters_padded_downscaled_*), or 'binary' (clusters_padded_binary_*, "
                        "0/100) (default %s)" % DATASET)
    p.add_argument("--total", type=int, default=TOTAL_IMAGES,
                   help="total clusters for the train/val/test split; train = 50%% of this (default %d)" % TOTAL_IMAGES)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="epochs (default %d)" % EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="mini-batch size (default %d)" % BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED, help="subset/split/shuffle seed (default %d)" % SEED)
    p.add_argument("--tile-tag", type=str, default=TILE_TAG, help="clustering tile size tag (default %s)" % TILE_TAG)
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                   help="folder with clusters_padded_<pct>_<tag>.h5 (default: clusterImages)")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help="where to save models (default: main_project/MBOONE/savedModel)")
    p.add_argument("--threads", type=str, default=str(CPU_THREADS),
                   help="CPU worker threads: 'auto' (half the cores) or an int (default %s). "
                        "Applied before the TF import." % CPU_THREADS)
    p.add_argument("--priority", choices=["low", "normal"],
                   default=("low" if LOW_PRIORITY else "normal"),
                   help="OS process priority (default %s); 'low' = below-normal so the machine "
                        "stays responsive for other work" % ("low" if LOW_PRIORITY else "normal"))
    p.add_argument("--self-test", action="store_true",
                   help="architecture checks only (no data, no training) and exit")
    a = p.parse_args(argv)
    a.data_dir = Path(a.data_dir)
    a.output_dir = Path(a.output_dir)
    a.percentiles = [int(v) for v in str(a.percentile).split(",") if v.strip()]
    return a


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return
    _check_dataset(args.dataset)                # crop | downscale, else abort clearly
    print("DATASET: %s  -> %s" % (args.dataset,
          DATASET_FILES[args.dataset].replace("%d", "<pct>") % args.tile_tag))
    if args.priority == "low":
        print("OS priority: below-normal (%s)" % ("set" if _set_low_priority() else "could not set"))
    print("CPU threads: %d of %d logical cores  (edit CPU_THREADS / --threads / CLUSTERING_THREADS)"
          % (_THREADS, os.cpu_count() or 0))
    for pct in args.percentiles:
        if pct not in (50, 75):
            raise SystemExit("unsupported percentile %d (expected 50 or 75)" % pct)
        train_one_percentile(pct, args)


if __name__ == "__main__":
    main()
