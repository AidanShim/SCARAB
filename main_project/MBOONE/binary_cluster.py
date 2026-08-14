# Binary-charge cluster images: take an existing padded cluster set (default the 75th
# percentile, clusters_padded_75_<tag>.h5) and rewrite it with the charge MAGNITUDE
# removed -- every pixel with any charge (> 0) becomes CHARGE_SET (100), everything else
# stays 0.  The images keep the SAME shape, ordering, and metadata as the source, so the
# result is a drop-in training/eval set; only the pixel values change (now two-valued:
# 0 or 100).
#
# WHY: to test whether the interaction-level autoencoder still flags anomalies when charge
# is NOT a factor -- i.e. when the input encodes only occupancy (charge / no-charge),
# not intensity.  Comparing this against the normal (crop) run isolates how much of the
# anomaly signal was really topology vs. just "more/brighter charge".
#
# OUTPUT (new file, never overwrites the source):
#   clusterImages/clusters_padded_binary_<pct>_<tag>.h5   (default pct=75)
# Train on it with:  python Clustering_Training.py --dataset binary --percentile 75
#
# RUN (repo root, ubopendata env):
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/binary_cluster.py
#   # only the 75th by default; override with BINARY_PCTS, the source variant with
#   # BINARY_SOURCE=crop|downscale, the "on" value with BINARY_CHARGE_SET, force overwrite
#   # with BINARY_FORCE=1.

import os
from pathlib import Path

import numpy as np
import h5py

PLANE = 2

# ------------------------------- Configuration -------------------------------
SIZE_TAG = os.environ.get("TILE_CLUSTER_SIZE_TAG", "18x16")     # tile size the padded set was made with
# Which percentile padded file(s) to binarize.  The user only wants the 75th (larger crop,
# captures more of the interaction), so that is the default; add more with BINARY_PCTS=50,75.
PERCENTILES = [int(x) for x in os.environ.get("BINARY_PCTS", "75").split(",") if x.strip()]
# Which padded set to read the pixels from: "crop" (clusters_padded_<pct>) or "downscale"
# (clusters_padded_downscaled_<pct>).  Binarizing is the same operation either way.
SOURCE = os.environ.get("BINARY_SOURCE", "crop").lower()
CHARGE_SET = float(os.environ.get("BINARY_CHARGE_SET", "100"))  # value every charged pixel becomes
FORCE = bool(int(os.environ.get("BINARY_FORCE", "0")))
CHUNK = int(os.environ.get("BINARY_CHUNK", "1000"))            # rows held in RAM at once
GZIP_LEVEL = 4
STORE_DTYPE = np.float32

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("TILE_CLUSTER_OUT", SCRIPT_DIR / "clusterImages"))

SOURCE_FILES = {"crop": "clusters_padded_%d_%s.h5",
                "downscale": "clusters_padded_downscaled_%d_%s.h5"}
# -----------------------------------------------------------------------------


def src_name(pct):
    if SOURCE not in SOURCE_FILES:
        raise SystemExit("BINARY_SOURCE must be one of %s, got %r" % (sorted(SOURCE_FILES), SOURCE))
    return SOURCE_FILES[SOURCE] % (pct, SIZE_TAG)


def dst_name(pct):
    return "clusters_padded_binary_%d_%s.h5" % (pct, SIZE_TAG)


def binarize_file(pct):
    src = OUT_DIR / src_name(pct)
    dst = OUT_DIR / dst_name(pct)
    if not src.exists():
        raise SystemExit("source padded file not found: %s\n(make it first with tile_cluster.py)" % src)
    if dst.exists() and not FORCE:
        raise SystemExit("binary file exists: %s (BINARY_FORCE=1 to overwrite)" % dst)

    with h5py.File(str(src), "r") as fin:
        imgs_in = fin["images"]
        n, sw, st = imgs_in.shape
        print("BINARIZE %s -> %s   (%d images, %d x %d, charged pixel -> %g)"
              % (src.name, dst.name, n, sw, st, CHARGE_SET))

        with h5py.File(str(dst), "w") as fout:
            # Carry over every source attribute so the file stays a drop-in, then mark it binary.
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["binary"] = True
            fout.attrs["charge_set"] = CHARGE_SET
            fout.attrs["binarized_from"] = src.name
            fout.attrs["source_variant"] = SOURCE

            imgs_out = fout.create_dataset("images", shape=(n, sw, st), dtype=STORE_DTYPE,
                                           chunks=(1, sw, st), compression="gzip",
                                           compression_opts=GZIP_LEVEL, shuffle=True)

            nonzero_sum = 0.0
            for s in range(0, n, CHUNK):
                block = imgs_in[s:s + CHUNK]
                # every pixel with ANY charge (> 0) -> CHARGE_SET; the rest stay 0.
                out = np.where(block > 0, CHARGE_SET, 0.0).astype(STORE_DTYPE)
                imgs_out[s:s + out.shape[0]] = out
                nonzero_sum += float((out > 0).sum())
                if (s // CHUNK) % 20 == 0 or s + CHUNK >= n:
                    print("    %d / %d" % (min(s + CHUNK, n), n), flush=True)

            # Copy the metadata group verbatim (same per-cluster rows, same order).
            if "metadata" in fin:
                fin.copy("metadata", fout)

    print("done -> %s   (mean charged-pixel fraction: %.4f)"
          % (dst, nonzero_sum / (n * sw * st) if n else 0.0))
    return dst


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Source variant: %s   Percentile(s): %s   Out dir: %s"
          % (SOURCE, PERCENTILES, OUT_DIR))
    for pct in PERCENTILES:
        binarize_file(pct)


if __name__ == "__main__":
    main()
