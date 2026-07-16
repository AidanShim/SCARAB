"""Unsupervised Teacher convolutional autoencoder for MicroBooNE collection-plane
image segments (arXiv:2509.21817, Fig. 3a/3b; RAD4LArTPC models.py / 02_train_teacher.py).

One self-contained, CPU-only training script: it defines the Teacher autoencoder,
trains it on preprocessed .npy segment files (target == input), saves the best model,
and provides per-segment anomaly-score functions.

Intentional deviations from the reference implementation (documented, not hidden):
  * Parametrized to the input segment size, chosen by the SEGMENT_WIRE / SEGMENT_TIME
    knob near the top of this file (18x16 / 64x32 / 864x64), instead of the reference's
    hardcoded 18x16. For an (18,16,1) input it reproduces the reference model exactly
    (367,201 params) and passes every self-test below. (The reference models.py was
    hardcoded to 18x16 despite its stale "864X64" comment.)
  * The default --output filename derives from the segment size (e.g. teacher_18x16.keras)
    so training different slices does not overwrite one model. The reference saved a single
    ./savedModel/teacher_40X192 directory.
  * `epochs` defaults to 20. The reference used epochs=int(h_split*v_split*train_ratio)
    = 3840, an epoch count derived from a tiling parameter -- meaningless, and days on CPU.
  * A fixed random_state (--seed, default 42) is passed to train_test_split so the
    50/10/40 split and the held-out test set are reproducible. The reference omits it.
  * No tf.distribute.MirroredStrategy / GPU / device-placement logic. CPU only
    (GPUs are hidden via CUDA_VISIBLE_DEVICES before TensorFlow is imported).
  * No qkeras, Student network, knowledge distillation, or quantization (Teacher only).
  * No StandardScaler / normalization of any kind -- the anomaly score lives on the raw
    thresholded ADC scale, and normalizing silently breaks the method.
  * Saves the best model in the legacy single-file .h5 format (load_model-compatible).
    The .keras zip format intermittently fails to write large models (~0.8 GB, e.g. the
    864x64 Teacher with optimizer state) under h5py 3.x -- "I/O operation on closed file"
    from H5FD_fileobj_truncate, because the weights are streamed into the zip's file
    object. .h5 writes a real file and avoids it. (Reference saved a SavedModel directory.)

Pick the wire x time slice with the SEGMENT_WIRE / SEGMENT_TIME knob near the top of this file.

Usage:
  python training.py --input "inputData/bnb_WithWire_00_*_18x16.npy" --epochs 20
  python training.py --self-test          # architecture checks only, no data / training
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU only: hide GPUs before TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")    # quieter TensorFlow logging

import argparse
import gc
import glob
import time
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import Model

# =====================================================================================
# WIRE x TIME SLICE TO TRAIN ON.  <-- change these two numbers to pick the segment size.
# Supported: (18, 16), (64, 32), (864, 64). The .npy file(s) you pass to --input must
# hold segments of exactly this (wire, time) shape -- produce them from the processed
# image file with mod_data_loader.export_npy (or WireImages.segments).
# =====================================================================================
SEGMENT_WIRE = 864
SEGMENT_TIME = 64
SEGMENT_SHAPE = (SEGMENT_WIRE, SEGMENT_TIME, 1)

# Ordered layer names, per build spec section 4 (the Input layer counts as layer 1).
EXPECTED_LAYER_NAMES = [
    "teacher_inputs_", "teacher_reshape", "teacher_conv2d_1", "teacher_relu_1",
    "teacher_pool_1", "teacher_conv2d_2", "teacher_relu_2", "teacher_flatten",
    "teacher_latent", "teacher_dense", "teacher_reshape2", "teacher_relu_3",
    "teacher_conv2d_3", "teacher_relu_4", "teacher_upsampling", "teacher_conv2d_4",
    "teacher_relu_5", "teacher_outputs",
]

# Teacher parameter counts per segment size (paper Table 1); verified by --self-test.
KNOWN_PARAM_COUNTS = {(18, 16, 1): 367201, (64, 32, 1): 2492401, (864, 64, 1): 66789361}

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.5, 0.1, 0.4


def build_teacher(segment_shape: Tuple[int, int, int]) -> keras.Model:
    """Build the Teacher autoencoder as one continuous functional graph (encoder+decoder).

    ``segment_shape`` is ``(wires, time, 1)``. Internal shapes are derived from it, so the
    same 18-layer architecture serves 18x16, 64x32 and 864x64. Both spatial dims must be
    even: AveragePooling2D((2,2)) then UpSampling2D((2,2)) has to round-trip to the input
    size for the pixel-wise reconstruction loss. Every Conv2D is linear + a separate
    Activation("relu"); only the output Conv2D folds in activation="relu" (charge >= 0).
    """
    W, H, C = segment_shape
    if W % 2 or H % 2:
        raise ValueError(
            "segment spatial dims must be even for the (2,2) pool/upsample round-trip; "
            "got %r" % (segment_shape,)
        )
    w2, h2 = W // 2, H // 2

    inputs = Input(shape=segment_shape, name="teacher_inputs_")
    x = Reshape((W, H, C), name="teacher_reshape")(inputs)          # no-op, kept for fidelity
    x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
    x = Activation("relu", name="teacher_relu_1")(x)
    x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)          # average, not max
    x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
    x = Activation("relu", name="teacher_relu_2")(x)
    x = Flatten(name="teacher_flatten")(x)
    x = Dense(80, activation="relu", name="teacher_latent")(x)      # bottleneck
    x = Dense(w2 * h2 * 30, name="teacher_dense")(x)               # 9*8*30 = 2160 at 18x16
    x = Reshape((w2, h2, 30), name="teacher_reshape2")(x)          # (9, 8, 30) at 18x16
    x = Activation("relu", name="teacher_relu_3")(x)
    x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
    x = Activation("relu", name="teacher_relu_4")(x)
    x = UpSampling2D((2, 2), name="teacher_upsampling")(x)          # restores (W, H)
    x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
    x = Activation("relu", name="teacher_relu_5")(x)
    outputs = Conv2D(
        1, (3, 3), activation="relu", strides=1, padding="same", name="teacher_outputs"
    )(x)
    return Model(inputs, outputs, name="Teacher")


def total_pixel_intensity(x: np.ndarray) -> np.ndarray:
    """Sum of pixel values in each segment -> shape (N,)."""
    return np.sum(x, axis=(1, 2, 3))


def anomaly_score(model: keras.Model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Per-segment reconstruction MSE -> shape (N,); higher = more anomalous.

    The same quantity as the training loss, but evaluated per segment (mean over pixels,
    axes 1..3) instead of averaged over the batch.
    """
    recon = model.predict(x, batch_size=batch_size, verbose=0)
    return np.mean((x - recon) ** 2, axis=(1, 2, 3))


def scaled_anomaly_score(model: keras.Model, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """anomaly_score / total_pixel_intensity -> shape (N,) (paper section 3.1.5).

    Dividing out total intensity removes the trivial correlation between reconstruction
    error and raw signal strength, isolating sensitivity to topology. Empty segments
    (total intensity == 0) return 0.0 (not NaN).
    """
    score = anomaly_score(model, x, batch_size=batch_size)
    total = total_pixel_intensity(x)
    out = np.zeros_like(score)
    nonzero = total > 0
    out[nonzero] = score[nonzero] / total[nonzero]
    return out


class EpochTimer(keras.callbacks.Callback):
    """Print a wall-clock timing line per epoch (build spec section 5, 'CPU expectations')."""

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        self._t0 = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        print(
            "[epoch %d] %.1fs  loss=%.6g  val_loss=%.6g"
            % (epoch + 1, time.time() - self._t0,
               logs.get("loss", float("nan")), logs.get("val_loss", float("nan")))
        )


def peek_segment_shape(npy_path: str) -> Tuple[int, int, int]:
    """Read the (W, H, 1) segment shape from a .npy header without loading the array."""
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 4 or arr.shape[-1] != 1:
        raise ValueError(
            "expected an (N, W, H, 1) array in %s, got shape %r" % (npy_path, arr.shape)
        )
    return (int(arr.shape[1]), int(arr.shape[2]), 1)


def resolve_save_path(path: str) -> str:
    """Use the robust legacy .h5 whole-model format for the checkpoint.

    The .keras (zip) format intermittently fails to save large models (~0.8 GB, e.g. the
    864x64 Teacher with optimizer state) under h5py 3.x -- 'ValueError: I/O operation on
    closed file' from H5FD_fileobj_truncate, because the weights HDF5 is streamed into the
    zip's file object. Legacy .h5 writes to a real file (no fileobj driver) and avoids it;
    keras.models.load_model() reads .h5 the same way.
    """
    if path.endswith(".keras"):
        h5 = path[: -len(".keras")] + ".h5"
        print("NOTE: saving as %s (not %s): the .keras format can fail on large models "
              "with this h5py version." % (os.path.basename(h5), os.path.basename(path)))
        return h5
    return path


def run_self_test(segment_shape: Tuple[int, int, int]) -> None:
    """Architecture acceptance checks (build spec section 8). No data, no training."""
    W, H, C = segment_shape
    model = build_teacher(segment_shape)
    print("== self-test for segment shape %r ==" % (segment_shape,))
    model.summary()

    # 1. shape round-trip (this is the check that catches a missing UpSampling2D)
    out = model.predict(np.zeros((1, W, H, C), dtype="float32"), verbose=0)
    assert out.shape == (1, W, H, C), "shape round-trip failed: got %r" % (out.shape,)
    print("[ok] shape round-trip (1, %d, %d, 1) -> %r" % (W, H, out.shape))

    # 2. layer inventory + count
    names = [layer.name for layer in model.layers]
    assert len(names) == 18, "expected 18 layers, got %d" % len(names)
    assert names == EXPECTED_LAYER_NAMES, "layer names differ from spec section 4:\n%r" % (names,)
    print("[ok] 18 layers, ordered names match spec section 4")

    # 3. parameter count
    n = model.count_params()
    expected = KNOWN_PARAM_COUNTS.get(segment_shape)
    if expected is not None:
        assert n == expected, (
            "parameter count %d != expected %d for %r -- architecture is wrong"
            % (n, expected, segment_shape)
        )
        print("[ok] %d parameters (matches paper Table 1)" % n)
    else:
        print("[info] %d parameters (no reference count on record for %r)" % (n, segment_shape))

    # 4. non-negativity of the reconstruction (output activation is ReLU)
    r = model.predict((np.random.rand(8, W, H, C) * 100.0).astype("float32"), verbose=0)
    assert (r >= 0).all(), "reconstruction contains negative pixels"
    print("[ok] all reconstructed pixels >= 0")
    print("SELF-TEST PASSED")


def load_segments(npy_path: str, limit: Optional[int], drop_empty: bool,
                  segment_shape: Tuple[int, int, int]) -> np.ndarray:
    """Load one .npy segment file as float32, optionally dropping all-zero padding
    segments and/or subsampling to the first ``limit`` segments. No normalization."""
    X = np.load(npy_path).astype("float32", copy=False)
    if X.shape[1:] != segment_shape:
        raise ValueError(
            "segment shape mismatch in %s: %r vs expected %r -- set SEGMENT_WIRE/SEGMENT_TIME "
            "to match, or pass matching --input" % (npy_path, X.shape[1:], segment_shape)
        )
    if drop_empty:
        keep = X.reshape(X.shape[0], -1).any(axis=1)
        dropped = int((~keep).sum())
        X = X[keep]
        print("  --drop-empty: dropped %d all-zero padding segments, %d remain"
              % (dropped, X.shape[0]))
    if limit is not None:
        used = min(limit, X.shape[0])
        print("  --limit: using the first %d of %d segments" % (used, X.shape[0]))
        X = X[:limit]
    return X


def train(args: argparse.Namespace) -> None:
    """Load segment files, train the Teacher (target == input), checkpoint the best model."""
    files = sorted(glob.glob(args.input)) if args.input else []
    if not files:
        raise SystemExit("no .npy files matched --input %r" % args.input)

    first_shape = peek_segment_shape(files[0])
    if first_shape != SEGMENT_SHAPE:
        raise SystemExit(
            "SEGMENT_WIRE/SEGMENT_TIME = %r but %s holds %r segments -- edit the knob at "
            "the top of this file, or pass matching --input." % (SEGMENT_SHAPE, files[0], first_shape)
        )
    print("training on %r segments  |  %d input file(s)" % (SEGMENT_SHAPE, len(files)))

    model = build_teacher(SEGMENT_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    output_path = resolve_save_path(args.output)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.splitext(output_path)[0] + "_history.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)  # fresh history per run; appended across files within a run

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.CSVLogger(csv_path, append=True),
        EpochTimer(),
    ]

    # Reference used epochs = int(h_split * v_split * train_ratio) = 3840; we use --epochs (20).
    last_test = None
    for i, fp in enumerate(files):
        print("\n=== file %d/%d: %s ===" % (i + 1, len(files), fp))
        X = load_segments(fp, args.limit, args.drop_empty, SEGMENT_SHAPE)

        # 50 / 10 / 40 train / val / test, reproducible via --seed.
        X_tv, X_test = train_test_split(X, test_size=TEST_RATIO, random_state=args.seed)
        X_train, X_val = train_test_split(
            X_tv, test_size=VAL_RATIO / (VAL_RATIO + TRAIN_RATIO), random_state=args.seed
        )
        print("  split: train=%d  val=%d  test=%d" % (len(X_train), len(X_val), len(X_test)))

        model.fit(
            X_train, X_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            verbose=2,
        )

        last_test = X_test
        del X, X_tv, X_train, X_val
        gc.collect()

    print("\nbest model (min val_loss) saved to: %s" % output_path)
    print("loss history: %s" % csv_path)

    # Post-training score sanity (build spec section 8.5): expect a spike near zero + long tail.
    if last_test is not None and len(last_test):
        s = anomaly_score(model, last_test)
        median, p90, p99, mx = np.percentile(s, [50, 90, 99, 100])
        print(
            "anomaly score on held-out test (n=%d): median=%.4g  p90=%.4g  p99=%.4g  max=%.4g"
            % (len(last_test), median, p90, p99, mx)
        )
        print("(expect the median near 0 with a long tail to high values; a broad or flat "
              "distribution usually means the pixel values got normalized somewhere)")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the unsupervised Teacher autoencoder (CPU, single file)."
    )
    p.add_argument("--input", type=str, default=None,
                   help="glob for preprocessed .npy segment file(s), each shape (N, W, H, 1)")
    p.add_argument("--epochs", type=int, default=20,
                   help="training epochs (reference used 3840; see module docstring). Default 20")
    p.add_argument("--batch-size", type=int, default=128, help="mini-batch size (default 128)")
    p.add_argument("--output", type=str,
                   default="./savedModel/teacher_%dx%d.h5" % (SEGMENT_WIRE, SEGMENT_TIME),
                   help="path to save the best model (.h5). Default derives from the segment size")
    p.add_argument("--seed", type=int, default=42,
                   help="random_state for the train/val/test split (default 42)")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N segments per file (fast smoke tests). Default: all")
    p.add_argument("--drop-empty", action="store_true",
                   help="drop all-zero padding segments before splitting/training (default off)")
    p.add_argument("--self-test", action="store_true",
                   help="run architecture acceptance checks (no data, no training) and exit")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test(SEGMENT_SHAPE)
        return
    train(args)


if __name__ == "__main__":
    main()
