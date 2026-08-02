# Batch-preprocess every MicroBooNE bnb_WithWire_*.h5 file into a compact,
# training-ready HDF5 file of collection-plane (plane 2) images.
#
# This is the multi-file version of tutorial/MBOONE/mod_data.py. It walks over
# ALL bnb_WithWire_{index}.h5 files in data/ubopendata/ and, for each, writes a
# plane-2-only, time-downsampled image file into data/ubopendata/processed_npy/.
#
# Follows the preprocessing of Chung et al., "Real-time Anomaly Detection for
# Liquid Argon Time Projection Chambers" (RAD4LArTPC / FIREFLY): plane 2 only,
# time axis compressed by 10 (summing every 10 ticks: 6400 -> 640).
#
# WHAT IS STORED (and what is NOT):
#   Per input file we store ONE array: the downsampled, NO-CUT plane-2 image per
#   event, in a gzip-compressed + per-event-chunked HDF5 dataset "image" of shape
#   (N, 3456, 640). No pixel cuts and no segment tiling are baked in -- those are
#   cheaply derived at load time (see tutorial/MBOONE/mod_data_loader.py):
#     - pixel cutoff / saturation cuts  -> applied per pixel on load
#     - 864x64 / 64x32 / 18x16 tilings  -> a reshape on load
#   Cutting-then-tiling == tiling-then-cutting, so nothing is lost by storing the
#   uncut image, and the cut values can be retuned later without reprocessing.
#
# WHY THIS FORMAT: LArTPC images are ~99.6% zeros. Stored densely as float32 the
# raw tiled .npy files were tens of GB; gzip-chunked HDF5 shrinks each file to
# ~0.1 GB while keeping lazy per-event random access (no loading it all into RAM).
#
# NOTE ON READING: we read wire_table directly with h5py instead of
# pynuml.io.File.read_data(), which in this env has a bug (only returns data for
# start==0, silently yielding all-zero events for later batches).
#
# NOTE ON OUTPUT FOLDER NAME: the destination folder is called "processed_npy"
# for historical reasons; the files written here are actually .h5, not .npy.
#
# RUN (from the SCARAB repo root) with the ubopendata env active:
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/plane_2.py
# Process a subset of files (inclusive, zero-padded index range):
#   MOD_DATA_FILE_START=0 MOD_DATA_FILE_END=5 conda run --no-capture-output -n ubopendata python main_project/MBOONE/plane_2.py
# Quick test on a few events per file:
#   MOD_DATA_N_EVENTS=8 conda run --no-capture-output -n ubopendata python main_project/MBOONE/plane_2.py
# Re-process files whose output already exists:
#   MOD_DATA_FORCE=1 conda run --no-capture-output -n ubopendata python main_project/MBOONE/plane_2.py

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

# float32 keeps an EXACT round-trip; after gzip the size cost over float16 is
# negligible, so we keep float32 for correctness.
STORE_DTYPE = np.float32
GZIP_LEVEL = 4            # gzip compression level (source file also uses gzip)

# Events whose wire rows are read from the HDF5 file at once (~0.2 GB/event).
READ_BATCH = 4

# Which bnb_WithWire files to process (inclusive index range). 0..17 by default.
FILE_START = int(os.environ.get("MOD_DATA_FILE_START", 0))
FILE_END = int(os.environ.get("MOD_DATA_FILE_END", 17))

# Re-process a file even if its output already exists (default: skip existing).
FORCE = bool(int(os.environ.get("MOD_DATA_FORCE", "0")))

# How many events to process per file. None = all. Override with MOD_DATA_N_EVENTS.
N_EVENTS = None
if os.environ.get("MOD_DATA_N_EVENTS"):
    N_EVENTS = int(os.environ["MOD_DATA_N_EVENTS"])

# Paths resolved relative to this script. Layout: SCARAB/main_project/MBOONE/plane_2.py
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_DIR = REPO_ROOT / "data" / "ubopendata"
OUTPUT_DIR = Path(os.environ.get("MOD_DATA_OUT", INPUT_DIR / "processed_npy"))
# -------------------------------------------------------------------------


def extract_plane(adc_rows, plane_rows):
    """Select the collection-plane rows from one event's wire block -> (nwires(PLANE), ntimeticks())."""
    return adc_rows[plane_rows == PLANE]


def downsample_time(img):
    """Sum every TIME_FACTOR ticks along the time axis: (W, 6400) -> (W, 640)."""
    return block_reduce(img, block_size=(1, TIME_FACTOR), func=np.sum)


def process_file(input_file, output_dir):
    """Convert one bnb_WithWire_*.h5 into a plane-2, time-downsampled image .h5.

    Returns the output Path on success, or None if skipped.
    """
    W = nwires(PLANE)                 # 3456
    T = ntimeticks() // TIME_FACTOR   # 640

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem  # e.g. "bnb_WithWire_00"
    out_path = output_dir / ("%s_plane%d_images.h5" % (stem, PLANE))

    if out_path.exists() and not FORCE:
        print("SKIP  %s (output exists; set MOD_DATA_FORCE=1 to overwrite)" % out_path.name)
        return None

    with h5py.File(str(input_file), "r") as h5, h5py.File(str(out_path), "w") as out:
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
        dset.attrs["source_file"] = input_file.name
        dset.attrs["content"] = "downsampled no-cut plane-%d ADC-sum images (wire, time)" % PLANE

        print("Input : %s (%d events in file)" % (input_file.name, n_file))
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
    return out_path


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError("Input directory not found: %s" % INPUT_DIR)

    print("Input dir : %s" % INPUT_DIR)
    print("Output dir: %s" % OUTPUT_DIR)
    print("Files     : %02d..%02d  (force=%s, n_events=%s)\n"
          % (FILE_START, FILE_END, FORCE, N_EVENTS))

    processed, skipped, missing, failed = [], [], [], []
    for filenum in range(FILE_START, FILE_END + 1):
        input_file = INPUT_DIR / ("bnb_WithWire_%02d.h5" % filenum)
        if not input_file.exists():
            print("MISS  %s (not found, skipping)" % input_file.name)
            missing.append(filenum)
            continue
        try:
            out_path = process_file(input_file, OUTPUT_DIR)
            (processed if out_path is not None else skipped).append(filenum)
        except Exception as exc:  # don't let one bad file abort the whole batch
            print("FAIL  %s: %s" % (input_file.name, exc))
            failed.append(filenum)
        print()

    print("=" * 60)
    print("Batch complete.")
    print("  processed: %s" % processed)
    print("  skipped  : %s (output already existed)" % skipped)
    print("  missing  : %s (input not found)" % missing)
    print("  failed   : %s" % failed)


if __name__ == "__main__":
    main()
