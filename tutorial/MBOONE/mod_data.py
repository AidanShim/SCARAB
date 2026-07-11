# Preprocess a MicroBooNE bnb_WithWire_*.h5 file into a compact, training-ready
# HDF5 file of collection-plane (plane 2) images.
#
# Follows the preprocessing of Chung et al., "Real-time Anomaly Detection for
# Liquid Argon Time Projection Chambers" (RAD4LArTPC / FIREFLY): plane 2 only,
# time axis compressed by 10 (summing).
#
# WHAT IS STORED (and what is NOT):
#   We store ONE array: the downsampled, NO-CUT plane-2 image per event, in a
#   gzip-compressed + per-event-chunked HDF5 dataset "image" of shape
#   (N, 3456, 640). Everything the older four .npy files held is cheaply
#   derived from this at load time (see mod_data_loader.py):
#     - pixel cutoff / saturation cuts  -> applied per pixel on load
#     - 864x64 / 64x32 / 18x16 tilings  -> a reshape on load
#   Cutting-then-tiling == tiling-then-cutting, so nothing is lost by storing the
#   uncut image, and you can retune the cut values later without reprocessing.
#
# WHY THIS FORMAT: LArTPC images are ~99.6% zeros. Stored densely as float32 the
# four .npy files were ~38 GB; the *source* HDF5 is only ~2 GB because it is
# gzip-chunked. This mirrors that: ~38 GB -> ~0.1-0.3 GB, still with lazy
# per-event random access (no need to load it all into RAM).
#
# NOTE ON READING: we read wire_table directly with h5py instead of
# pynuml.io.File.read_data(), which in this env has a bug (only returns data for
# start==0, silently yielding all-zero events for later batches).
#
# RUN (from the SCARAB repo root) with the ubopendata env active:
#   conda run --no-capture-output -n ubopendata python tutorial/MBOONE/mod_data.py
# Quick test on a few events:
#   MOD_DATA_N_EVENTS=8 conda run --no-capture-output -n ubopendata python tutorial/MBOONE/mod_data.py

import os
import gc
from pathlib import Path

import numpy as np
import h5py
from skimage.measure import block_reduce

from microboone_utils import nwires, ntimeticks  # detector geometry helpers

# ----------------------------- Configuration -----------------------------
PLANE = 2                 # collection plane (the one Chung et al. use)
TIME_FACTOR = 10          # compress the time axis by summing every 10 ticks (6400 -> 640)

# Default cut values, saved as metadata and applied by the loader (not baked in).
DEFAULT_CUTOFF = 10       # summed-ADC below this -> 0 (noise removal). Paper value.
DEFAULT_SATURATION = 100  # summed-ADC above this -> clipped. Paper value.

# Segment sizes (wire, time) the loader can tile into. Saved as a hint.
SEGMENT_SIZES = [(864, 64), (64, 32), (18, 16)]

# float32 keeps an EXACT round-trip. float16 would halve the (already tiny)
# uncompressed size but can flip pixels across the cutoff threshold by ~10 ADC;
# after gzip the size difference is negligible (~50 MB vs ~33 MB total), so we
# keep float32 for correctness.
STORE_DTYPE = np.float32
GZIP_LEVEL = 4            # gzip compression level (source file also uses gzip)

# Events whose wire rows are read from the HDF5 file at once (~0.2 GB/event).
READ_BATCH = 4

# How many events to process. None = all events. Override with MOD_DATA_N_EVENTS.
N_EVENTS = None
if os.environ.get("MOD_DATA_N_EVENTS"):
    N_EVENTS = int(os.environ["MOD_DATA_N_EVENTS"])

# Paths resolved relative to this script. Repo layout: SCARAB/tutorial/MBOONE/mod_data.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_FILE = REPO_ROOT / "data" / "ubopendata" / "bnb_WithWire_00.h5" # Adjust this for different files
OUTPUT_DIR = Path(os.environ.get("MOD_DATA_OUT", REPO_ROOT / "data" / "ubopendata" / "processed_npy"))
# -------------------------------------------------------------------------


def extract_plane(adc_rows, plane_rows):
    """Select the collection-plane rows from one event's wire block -> (nwires(PLANE), ntimeticks())."""
    return adc_rows[plane_rows == PLANE]


def downsample_time(img):
    """Sum every TIME_FACTOR ticks along the time axis: (W, 6400) -> (W, 640)."""
    return block_reduce(img, block_size=(1, TIME_FACTOR), func=np.sum)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input file not found: %s\n"
            "Put the bnb_WithWire HDF5 file there, or edit INPUT_FILE." % INPUT_FILE
        )

    W = nwires(PLANE)                 # 3456
    T = ntimeticks() // TIME_FACTOR   # 640

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = INPUT_FILE.stem  # e.g. "bnb_WithWire_00"
    out_path = OUTPUT_DIR / ("%s_plane%d_images.h5" % (stem, PLANE))

    with h5py.File(str(INPUT_FILE), "r") as h5, h5py.File(str(out_path), "w") as out:
        n_file = h5["event_table/event_id"].shape[0]
        n_total = n_file if N_EVENTS is None else min(N_EVENTS, n_file)

        # Row layout of the wire_table: event e occupies rows [offsets[e]:offsets[e+1]].
        seq_cnt = h5["wire_table/event_id.seq_cnt"][:]           # (n_file, 2)
        offsets = np.concatenate([[0], np.cumsum(seq_cnt[:, 1])])
        adc_ds = h5["wire_table/adc"]                            # (total_rows, 6400)
        plane_ds = h5["wire_table/local_plane"]                 # (total_rows, 1)

        # Output: one gzip-compressed, per-event-chunked dataset of no-cut images.
        dset = out.create_dataset(
            "image", shape=(n_total, W, T), dtype=STORE_DTYPE,
            chunks=(1, W, T), compression="gzip", compression_opts=GZIP_LEVEL, shuffle=True,
        )
        # Metadata the loader uses to reproduce cuts/tilings.
        dset.attrs["plane"] = PLANE
        dset.attrs["time_factor"] = TIME_FACTOR
        dset.attrs["default_cutoff"] = DEFAULT_CUTOFF
        dset.attrs["default_saturation"] = DEFAULT_SATURATION
        dset.attrs["segment_sizes"] = np.array(SEGMENT_SIZES, dtype=np.int32)
        dset.attrs["source_file"] = INPUT_FILE.name
        dset.attrs["content"] = "downsampled no-cut plane-%d ADC-sum images (wire, time)" % PLANE

        print("Input : %s (%d events in file)" % (INPUT_FILE.name, n_file))
        print("Events: processing %d, plane %d, time factor %d" % (n_total, PLANE, TIME_FACTOR))
        print("Output: %s  dataset 'image' %s %s (gzip-%d, chunk per event)"
              % (out_path.name, (n_total, W, T), np.dtype(STORE_DTYPE).name, GZIP_LEVEL))

        written = 0
        for start in range(0, n_total, READ_BATCH):
            count = min(READ_BATCH, n_total - start)
            lo, hi = int(offsets[start]), int(offsets[start + count])
            plane_block = plane_ds[lo:hi, 0]   # (rows,)
            adc_block = adc_ds[lo:hi, :]       # (rows, 6400) -- the heavy read

            r = 0
            for e in range(start, start + count):
                nrows = int(seq_cnt[e, 1])
                plane_e = plane_block[r:r + nrows]
                adc_e = adc_block[r:r + nrows, :]
                r += nrows

                p2 = extract_plane(adc_e, plane_e)
                if p2.shape[0] != W:
                    raise RuntimeError(
                        "event %d: expected %d plane-%d wires, got %d" % (e, W, PLANE, p2.shape[0])
                    )
                dset[written] = downsample_time(p2).astype(STORE_DTYPE)   # no cuts stored
                written += 1

            del adc_block, plane_block
            gc.collect()
            print("  processed %d / %d events" % (written, n_total), flush=True)

    size_gb = os.path.getsize(out_path) / 1e9
    print("Done. Wrote %d events -> %s (%.3f GB)" % (written, out_path, size_gb))


if __name__ == "__main__":
    main()
