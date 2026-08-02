# Compute per-tile statistics for the processed plane-2 image files, so that
# mostly-empty tiles can be filtered out of autoencoder training.
#
# WHY: LArTPC images are ~99.6% zeros. After tiling, the majority of tiles at the
# smaller segment sizes contain NO surviving charge at all (measured on a sample:
# 9% empty at 864x64, 84% at 64x32, 95% at 18x16). An autoencoder trained on that
# can minimize reconstruction loss by learning the degenerate "emit zeros"
# mapping, burning capacity without learning track topology.
#
# WHAT THIS STORES: for every (event, tile) we store two threshold-INDEPENDENT
# numbers, computed AFTER the pixel cuts are applied:
#   charge    - sum of cut pixel intensities in the tile (float32)
#   occupancy - number of non-zero pixels in the tile   (uint16)
# No threshold is baked in. Filtering is then a boolean mask at load time
# (e.g. occupancy >= 5), so the cut can be retuned without recomputing anything.
# This mirrors mod_data.py's philosophy of storing uncut data and deriving later.
#
# The charge column doubles as the denominator for the paper's "scaled anomaly
# score" (anomaly score / total pixel intensity, Chung et al. section 3.1.5).
#
# ORDER OF OPERATIONS MATTERS: cuts are applied BEFORE summing. On raw no-cut
# data every tile carries noise and the sums are meaningless for emptiness.
#
# TILE ORDERING matches mod_data_loader.split_into_segments (wire-tile major,
# time-tile fastest), so a tile index here indexes that function's output
# directly. This is asserted at startup by _check_ordering().
#
# RUN (from the SCARAB repo root) with the ubopendata env active:
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/tile_stats.py
# Subset / testing:
#   MOD_DATA_FILE_START=0 MOD_DATA_FILE_END=0 ... python main_project/MBOONE/tile_stats.py
#   TILE_STATS_N_EVENTS=16 ... python main_project/MBOONE/tile_stats.py
#   TILE_STATS_FORCE=1 ...   python main_project/MBOONE/tile_stats.py

import os
import gc
from pathlib import Path

import numpy as np
import h5py

# ----------------------------- Configuration -----------------------------
SEGMENT_SIZES = [(864, 64), (64, 32), (18, 16)]

# Pixel cuts applied before summing. Defaults match the values stored on the
# image dataset by plane_2.py (and the paper); overridden per-file by its attrs.
FALLBACK_CUTOFF = 10.0
FALLBACK_SATURATION = 100.0

GZIP_LEVEL = 4

FILE_START = int(os.environ.get("MOD_DATA_FILE_START", 0))
FILE_END = int(os.environ.get("MOD_DATA_FILE_END", 17))
FORCE = bool(int(os.environ.get("TILE_STATS_FORCE", "0")))

N_EVENTS = None
if os.environ.get("TILE_STATS_N_EVENTS"):
    N_EVENTS = int(os.environ["TILE_STATS_N_EVENTS"])

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROC_DIR = Path(os.environ.get("MOD_DATA_OUT",
                               REPO_ROOT / "data" / "ubopendata" / "processed_npy"))
# -------------------------------------------------------------------------


def apply_cuts(img, cutoff, saturation):
    """Zero sub-threshold pixels, clip saturated ones. Returns a float32 copy."""
    out = np.asarray(img, dtype=np.float32).copy()
    out[out < cutoff] = 0
    out[out > saturation] = saturation
    return out


def tile_stats(img, tw, tt):
    """Per-tile (charge_sum, occupancy) for a cut image.

    Returned flat in split_into_segments order: wire-tile major, time-tile fastest.
    """
    nv, nh = img.shape[0] // tw, img.shape[1] // tt
    v = img.reshape(nv, tw, nh, tt)
    charge = v.sum(axis=(1, 3), dtype=np.float64).reshape(-1)
    occ = (v != 0).sum(axis=(1, 3)).reshape(-1)
    return charge.astype(np.float32), occ.astype(np.uint16)


def _split_into_segments(img, seg_wire, seg_time):
    """Reference tiling from mod_data_loader.py, used only for the ordering check."""
    v_split = img.shape[0] // seg_wire
    h_split = img.shape[1] // seg_time
    X = np.array(np.split(img, h_split, axis=1))
    X = np.array(np.split(X, v_split, axis=1))
    return np.reshape(X, (-1, seg_wire, seg_time, 1))


def _check_ordering():
    """Fail fast if the vectorized tiling ever diverges from the loader's."""
    rng = np.random.default_rng(0)
    probe = (rng.random((3456, 640)) * 200).astype(np.float32)
    for tw, tt in SEGMENT_SIZES:
        ref = _split_into_segments(probe, tw, tt)[..., 0].sum(axis=(1, 2))
        mine, _ = tile_stats(probe, tw, tt)
        if ref.shape != mine.shape or not np.allclose(ref, mine, rtol=1e-4):
            raise RuntimeError("tile ordering mismatch vs split_into_segments "
                               "for %dx%d" % (tw, tt))


def process_file(image_path, out_path):
    """Write per-tile stats for one processed image file. Returns out_path or None."""
    if out_path.exists() and not FORCE:
        print("SKIP  %s (exists; set TILE_STATS_FORCE=1 to overwrite)" % out_path.name)
        return None

    with h5py.File(str(image_path), "r") as src, h5py.File(str(out_path), "w") as out:
        d = src["image"]
        n_file = d.shape[0]
        n_total = n_file if N_EVENTS is None else min(N_EVENTS, n_file)
        cutoff = float(d.attrs.get("default_cutoff", FALLBACK_CUTOFF))
        saturation = float(d.attrs.get("default_saturation", FALLBACK_SATURATION))

        out.attrs["source_file"] = image_path.name
        out.attrs["cutoff"] = cutoff
        out.attrs["saturation"] = saturation
        out.attrs["note"] = ("per-tile charge/occupancy computed AFTER cuts; "
                             "no emptiness threshold applied")

        dsets = {}
        for tw, tt in SEGMENT_SIZES:
            ntiles = (d.shape[1] // tw) * (d.shape[2] // tt)
            g = out.create_group("%dx%d" % (tw, tt))
            g.attrs["seg_wire"], g.attrs["seg_time"] = tw, tt
            g.attrs["tiles_per_event"] = ntiles
            g.attrs["tile_order"] = "wire-tile major, time-tile fastest (split_into_segments)"
            dsets[(tw, tt)] = (
                g.create_dataset("charge", shape=(n_total, ntiles), dtype=np.float32,
                                 compression="gzip", compression_opts=GZIP_LEVEL),
                g.create_dataset("occupancy", shape=(n_total, ntiles), dtype=np.uint16,
                                 compression="gzip", compression_opts=GZIP_LEVEL),
            )

        print("  %s: %d events, cuts=(%.0f, %.0f)"
              % (image_path.name, n_total, cutoff, saturation))

        for i in range(n_total):
            img = apply_cuts(d[i], cutoff, saturation)
            for s in SEGMENT_SIZES:
                charge, occ = tile_stats(img, *s)
                dsets[s][0][i] = charge
                dsets[s][1][i] = occ
            if (i + 1) % 200 == 0 or (i + 1) == n_total:
                print("    %d / %d events" % (i + 1, n_total), flush=True)
            del img
        gc.collect()

    return out_path


def main():
    _check_ordering()
    print("tile ordering matches split_into_segments for all sizes")
    print("Processed dir: %s" % PROC_DIR)
    print("Files: %02d..%02d (force=%s, n_events=%s)\n"
          % (FILE_START, FILE_END, FORCE, N_EVENTS))

    done, skipped, missing, failed = [], [], [], []
    for filenum in range(FILE_START, FILE_END + 1):
        image_path = PROC_DIR / ("bnb_WithWire_%02d_plane2_images.h5" % filenum)
        out_path = PROC_DIR / ("bnb_WithWire_%02d_tilestats.h5" % filenum)
        if not image_path.exists():
            print("MISS  %s (not found)" % image_path.name)
            missing.append(filenum)
            continue
        try:
            r = process_file(image_path, out_path)
            (done if r is not None else skipped).append(filenum)
        except Exception as exc:
            print("FAIL  %s: %s" % (image_path.name, exc))
            failed.append(filenum)

    print("\n" + "=" * 60)
    print("Tile stats complete.")
    print("  written : %s" % done)
    print("  skipped : %s" % skipped)
    print("  missing : %s" % missing)
    print("  failed  : %s" % failed)


if __name__ == "__main__":
    main()
