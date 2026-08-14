# SCARAB interaction-extraction pipeline: cluster non-empty tiles into individual
# particle-interaction images, then pad them to a consistent size for an
# (interaction-level) autoencoder.  Implements Stages 3-4 of the SCARAB Clustering
# & Padding Guide; Stages 1-2 (tiling + charge thresholding) are already done --
# the surviving-tile map is read straight from the nonEmptyImages2 set.
#
# WHERE THE INPUTS COME FROM
#   * active-tile map : main_project/MBOONE/nonEmptyImages2/
#                       bnb_WithWire_XX_plane2_<W>x<T>_nonempty.h5  (charge >= 50)
#                       Each event's kept tiles ARE the guide's binary map -- a tile
#                       is "active" iff it survived the >=50 cut.  We rebuild the
#                       (wire_blocks, time_blocks) binary grid from event_index /
#                       tile_index (same tile numbering as remove_empty /
#                       mod_data_loader.split_into_segments: flat = wireblk*time_blocks
#                       + timeblk).
#   * pixels          : data/ubopendata/processed_npy/bnb_WithWire_XX_plane2_images.h5
#                       The FULL processed image (dataset "image", (N, 3456, 640)),
#                       cut with the file's own (cutoff, saturation).  Extraction
#                       reads pixels from here (guide's "extract from the
#                       full-resolution image"), so faint 10-50 ADC charge inside a
#                       cluster's bounding box is preserved.  Set PIXEL_SOURCE=
#                       "reconstruct" to instead rebuild pixels from the >=50 tiles
#                       alone (self-contained; sub-50 tiles inside the bbox become 0).
#
# THE FOUR STAGES (guide sec. 1)
#   1 Tiling        - already done (nonEmptyImages2 tiles the whole 3456x640 image).
#   2 Thresholding  - already done (charge >= 50 -> "active").
#   3 Clustering    - king-move / 8-connectivity connected components on the binary
#                     tile grid (scipy.ndimage.label, 3x3 structure).  Clusters with
#                     < MIN_CLUSTER_TILES tiles are dropped as noise; clusters that
#                     span the detector (> MAX_CLUSTER_TILES tiles OR > MAX_SPAN_FRAC
#                     of the grid in either axis) are dropped as cosmics and reported.
#   4 Extract + pad - convert each cluster's tile bbox to pixel coords (+ PAD_TILES of
#                     padding), crop from the full image, and (pass 2) zero-pad every
#                     crop to a common target chosen from the size histogram.
#
# ORIENTATION / NAMING.  Images are stored (wire, time) = (3456, 640).  The tile grid
# rows are wire-blocks (axis 0), cols are time-blocks (axis 1).  To match the guide's
# physical convention ("18 wide = wires, 16 tall = time"), we report
#     image_width_px  == wire extent (axis 0)
#     image_height_px == time extent (axis 1)
# and the padding target is (TARGET_WIRE, TARGET_TIME) = (axis 0, axis 1).
#
# TILE SIZE IS CONFIGURABLE.  SIZE = (SEG_WIRE, SEG_TIME); everything (input file
# names, grid shape, output names) derives from it, so running 64x32 instead of 18x16
# is a one-line change (or TILE_CLUSTER_SIZE=64x32 in the env).  Deliverable file
# names embed the size so runs never overwrite each other.
#
# DELIVERABLES (guide sec. 0)
#   D1 clusters_raw_<W>x<T>.h5     - every extracted cluster (variable size) + full
#                                    per-cluster metadata.  [pass 1]  (the "analysis" file)
#   D2 cluster_size_distribution_<W>x<T>.png - 4-panel size histogram.  [pass 1]
#   D3 cluster_samples_<W>x<T>.png - 40 random extracted clusters.      [pass 1]
#   D4 clusters_per_event_<W>x<T>.png - clusters-per-event histogram.   [pass 1]
#   D5 cluster_summary_<W>x<T>.txt - printed + saved run summary.        [pass 1]
#   D6 clusters_padded_<pct>_<W>x<T>.h5 - clusters padded to the <pct>th-pct target +
#                                    metadata; ONE file per percentile (default 50, 75).
#                                    [pass 2]  (the "training" files the DataLoader reads)
#                                    Oversized clusters are CENTER-CROPPED to the target.
#   D7 clusters_padded_downscaled_<pct>_<W>x<T>.h5 - ALTERNATE training files [pass 3]:
#                                    SAME per-percentile targets as D6, but oversized
#                                    clusters are DOWNSCALED (area-averaged) to fit instead
#                                    of cropped -- topology kept, resolution sacrificed --
#                                    AND blip clusters (<= PASS3_MIN_TILES-1 tiles, default
#                                    <=4) are dropped.  Fewer rows than D6; metadata carries
#                                    a raw_index back to D1 (for truth-label lookup) plus
#                                    was_downscaled / downscale_factor.
#   D1/D6/D7 land in main_project/MBOONE/clusterImages/ ; D2-D5 in main_project/MBOONE/.
#
# RUN (from the SCARAB repo root, ubopendata env; numpy needs the env's Library/bin):
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/tile_cluster.py
#   # ^ pass 1: clusters + histograms + summary.  Look at D2/D5, then:
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/tile_cluster.py pass2
#   # ^ pass 2: writes one padded file per percentile in TILE_CLUSTER_PAD_PCTS
#   #   (default "50,75" -> clusters_padded_50_<size>.h5 and clusters_padded_75_<size>.h5).
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/tile_cluster.py pass3
#   # ^ pass 3 (ALTERNATE): same targets, but DOWNSCALES oversized clusters instead of
#   #   cropping and drops <=4-tile blips -> clusters_padded_downscaled_50/75_<size>.h5.
#   #   Tune the blip cut with TILE_CLUSTER_PASS3_MIN_TILES (default 5 = keep n_tiles>=5).
# Quick first look at one file / few events, and forced overwrite:
#   MOD_DATA_FILE_START=0 MOD_DATA_FILE_END=0 TILE_CLUSTER_N_EVENTS=200 \
#     conda run --no-capture-output -n ubopendata python main_project/MBOONE/tile_cluster.py
#   TILE_CLUSTER_FORCE=1 ... python main_project/MBOONE/tile_cluster.py
# Switch tile size / retune the cosmic cap:
#   TILE_CLUSTER_SIZE=64x32 ... ;  TILE_CLUSTER_MAX_TILES=500 TILE_CLUSTER_MAX_SPAN_FRAC=0.75 ...
# Architecture / logic check with no data:
#   python main_project/MBOONE/tile_cluster.py --self-test

import os
import sys
import json
from pathlib import Path

import numpy as np
import h5py
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")            # headless: we only ever save PNGs
import matplotlib.pyplot as plt

# ------------------------------- Configuration -------------------------------
PLANE = 2

# Tile size for this run (SEG_WIRE, SEG_TIME).  Change these two numbers (or set
# TILE_CLUSTER_SIZE=64x32) to process a different size; nothing else needs editing.
def _parse_size(s, default):
    if not s:
        return default
    w, t = s.lower().split("x")
    return int(w), int(t)

SEG_WIRE, SEG_TIME = _parse_size(os.environ.get("TILE_CLUSTER_SIZE"), (18, 16))

# Stage 3 clustering knobs.
MIN_CLUSTER_TILES = int(os.environ.get("TILE_CLUSTER_MIN_TILES", "3"))    # drop noise below this
# Cosmic caps (a cluster is "oversized" and DROPPED if it trips EITHER):
#   MAX_CLUSTER_TILES  - too many active tiles (merged cosmic mess).
#   MAX_SPAN_FRAC      - bbox spans > this fraction of the tile grid in either axis
#                        (catches a thin muon crossing the detector, which a tile-
#                        count cap misses).  Fraction -> auto-scales across tile sizes.
# Both are provisional: pass 1 reports the full size distribution INCLUDING the
# dropped giants (D5) so you can retune after the first run.  None disables a cap.
def _opt_int(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return None if v.lower() in ("", "none", "off") else int(v)

def _opt_float(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return None if v.lower() in ("", "none", "off") else float(v)

MAX_CLUSTER_TILES = _opt_int("TILE_CLUSTER_MAX_TILES", 300)
MAX_SPAN_FRAC = _opt_float("TILE_CLUSTER_MAX_SPAN_FRAC", 0.6)

# Stage 4 knob.
PAD_TILES = int(os.environ.get("TILE_CLUSTER_PAD", "1"))

# Where cluster pixels come from: "original" (processed_npy full image, guide-faithful)
# or "reconstruct" (rebuild from the >=50 tiles alone; sub-50 tiles -> 0).
PIXEL_SOURCE = os.environ.get("TILE_CLUSTER_PIXELS", "original").lower()

# Pass-2 padding: write one padded output file per percentile of the cluster
# (width, height) distribution.  Default 50th + 75th -> clusters_padded_50_<size>.h5
# and clusters_padded_75_<size>.h5.  Each target is that percentile of the per-cluster
# extents, rounded up to even.  (Pass 1 is unaffected by this.)
PAD_PERCENTILES = [int(x) for x in os.environ.get("TILE_CLUSTER_PAD_PCTS", "50,75").split(",") if x.strip()]

# Pass-3 (downscale) knob.  Pass 3 writes an ALTERNATE set of training files that
# (a) DOWNSCALE oversized clusters to the target instead of cropping them, and
# (b) drop small "blip" clusters with <= PASS3_MIN_TILES-1 tiles (default: <=4, i.e.
# keep only clusters with >= 5 tiles).  Pass 1/2 are unaffected.
PASS3_MIN_TILES = int(os.environ.get("TILE_CLUSTER_PASS3_MIN_TILES", "5"))    # keep n_tiles >= this

STORE_DTYPE = np.float32
GZIP_LEVEL = 4
N_SAMPLES = 40                  # clusters drawn for the D3 visual-check grid
SCATTER_MAX = 5000              # points plotted in the D2 scatter panel
FALLBACK_CUTOFF = 10.0
FALLBACK_SATURATION = 100.0

FILE_START = int(os.environ.get("MOD_DATA_FILE_START", 0))
FILE_END = int(os.environ.get("MOD_DATA_FILE_END", 17))
FORCE = bool(int(os.environ.get("TILE_CLUSTER_FORCE", "0")))
N_EVENTS = int(os.environ["TILE_CLUSTER_N_EVENTS"]) if os.environ.get("TILE_CLUSTER_N_EVENTS") else None
SEED = int(os.environ.get("TILE_CLUSTER_SEED", "0"))

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
IMG_DIR = Path(os.environ.get("MOD_DATA_OUT", REPO_ROOT / "data" / "ubopendata" / "processed_npy"))
TILE_DIR = Path(os.environ.get("TILE_CLUSTER_TILEDIR", SCRIPT_DIR / "nonEmptyImages2"))
OUT_DIR = Path(os.environ.get("TILE_CLUSTER_OUT", SCRIPT_DIR / "clusterImages"))
FIG_DIR = Path(os.environ.get("TILE_CLUSTER_FIGDIR", SCRIPT_DIR))     # D2-D5 -> main_project/MBOONE
# -----------------------------------------------------------------------------

SIZE_TAG = "%dx%d" % (SEG_WIRE, SEG_TIME)
RAW_NAME = "clusters_raw_%s.h5" % SIZE_TAG


def tile_name(filenum):
    return "bnb_WithWire_%02d_plane%d_%s_nonempty.h5" % (filenum, PLANE, SIZE_TAG)


def image_name(filenum):
    return "bnb_WithWire_%02d_plane%d_images.h5" % (filenum, PLANE)


def padded_name(pct):
    return "clusters_padded_%d_%s.h5" % (pct, SIZE_TAG)


def padded_downscaled_name(pct):
    return "clusters_padded_downscaled_%d_%s.h5" % (pct, SIZE_TAG)


def apply_cuts(img, cutoff, saturation):
    """Zero sub-threshold pixels, clip saturated ones. Returns a float32 copy."""
    out = np.asarray(img, dtype=np.float32).copy()
    out[out < cutoff] = 0
    out[out > saturation] = saturation
    return out


# ------------------------- Stage 3: king-move clustering ----------------------
_STRUCT8 = np.ones((3, 3), dtype=int)      # 8-connectivity (king moves)


def cluster_binary_map(binary_map, min_tiles=MIN_CLUSTER_TILES):
    """King-move (8-connectivity) connected components on a binary tile grid.

    Returns (clusters, discarded_small) where each cluster is a dict with
    cluster_id (ndimage label), n_tiles, and bbox_tile = (min_row, max_row,
    min_col, max_col) in (wire-block, time-block) coordinates.  Clusters with
    fewer than ``min_tiles`` tiles are dropped (noise) and counted.
    """
    labeled, n = ndimage.label(binary_map, structure=_STRUCT8)
    clusters, discarded_small = [], 0
    if n == 0:
        return clusters, discarded_small
    # Vectorized bbox + tile count per label (one pass, no python loop over tiles).
    objs = ndimage.find_objects(labeled)
    counts = np.bincount(labeled.ravel())      # counts[0] = background
    for cid, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        n_tiles = int(counts[cid])
        if n_tiles < min_tiles:
            discarded_small += 1
            continue
        rsl, csl = sl
        clusters.append({
            "cluster_id": cid,
            "n_tiles": n_tiles,
            "bbox_tile": (int(rsl.start), int(rsl.stop) - 1,
                          int(csl.start), int(csl.stop) - 1),
        })
    return clusters, discarded_small


def is_oversized(cluster, wire_blocks, time_blocks):
    """True if a cluster looks like a detector-spanning cosmic and should be dropped."""
    if MAX_CLUSTER_TILES is not None and cluster["n_tiles"] > MAX_CLUSTER_TILES:
        return True
    if MAX_SPAN_FRAC is not None:
        min_r, max_r, min_c, max_c = cluster["bbox_tile"]
        if (max_r - min_r + 1) > MAX_SPAN_FRAC * wire_blocks:
            return True
        if (max_c - min_c + 1) > MAX_SPAN_FRAC * time_blocks:
            return True
    return False


# ------------------------- Stage 4: extract + pad -----------------------------
def extract_cluster_image(full_image, bbox_tile, seg_wire=SEG_WIRE, seg_time=SEG_TIME,
                          pad_tiles=PAD_TILES):
    """Crop a cluster's (padded) bounding box from the full (wire, time) image.

    Returns (extracted, meta) with meta holding pixel-coord bbox, extent, and charge.
    width == wire extent (axis 0); height == time extent (axis 1).
    """
    min_row, max_row, min_col, max_col = bbox_tile
    W, T = full_image.shape
    py1 = max(0, (min_row - pad_tiles) * seg_wire)
    py2 = min(W, (max_row + 1 + pad_tiles) * seg_wire)
    px1 = max(0, (min_col - pad_tiles) * seg_time)
    px2 = min(T, (max_col + 1 + pad_tiles) * seg_time)
    extracted = full_image[py1:py2, px1:px2].copy()
    meta = {
        "bbox_pixel_coords": (py1, py2, px1, px2),
        "image_width_px": py2 - py1,      # wire extent
        "image_height_px": px2 - px1,     # time extent
        "total_charge": float(np.sum(np.abs(extracted))),
    }
    return extracted, meta


def pad_to_target(image, target_wire, target_time):
    """Center-crop (if larger) then center zero-pad to (target_wire, target_time)."""
    h, w = image.shape           # (wire, time)
    if h > target_wire or w > target_time:
        s0 = max(0, (h - target_wire) // 2)
        s1 = max(0, (w - target_time) // 2)
        image = image[s0:s0 + min(h, target_wire), s1:s1 + min(w, target_time)]
        h, w = image.shape
    out = np.zeros((target_wire, target_time), dtype=image.dtype)
    t0 = (target_wire - h) // 2
    t1 = (target_time - w) // 2
    out[t0:t0 + h, t1:t1 + w] = image
    return out


def _even_ceil(x):
    """Smallest even integer >= x (autoencoders need even spatial dims to pool/upsample)."""
    v = int(np.ceil(x))
    return v + 1 if v % 2 else v


# ---------------------- Stage 4 (pass 3): downscale instead of crop -----------
def _interp0(A, pos):
    """Linear-interpolate the ROWS of A (shape (n, ...)) at fractional positions pos (k,).

    Returns shape (k, ...).  Used to sample the summed-area table at fractional edges.
    """
    n = A.shape[0]
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 2)
    frac = (pos - i0).reshape((-1,) + (1,) * (A.ndim - 1))
    return A[i0] * (1.0 - frac) + A[i0 + 1] * frac


def area_resize(image, out_wire, out_time):
    """EXACT area-average resample of a (wire, time) image to (out_wire, out_time).

    Method: build the summed-area table (2-D cumulative sum) of the image.  Because
    the pixels are piecewise-constant, that integral is piecewise-BILINEAR, so
    linearly interpolating the table at the (fractional) output-cell edges gives the
    EXACT mean charge inside each output cell -- i.e. real area averaging / box
    downsampling, not point sampling.  Consequences:
      * Every input pixel contributes to some output cell, so a thin 1-px track can
        never fall "between samples" and disappear -- the track's PATH (its topology)
        is preserved; only resolution and fine detail are sacrificed, exactly the
        trade the request asks for.
      * It is mean-preserving (a flat patch keeps its value); the per-pixel charge
        SCALE is kept while the total sum scales with area -- and the scaled anomaly
        score (MSE / total charge) divides that out.
    Aspect ratio is the caller's responsibility (see downscale_to_target); this just
    maps h->out_wire, w->out_time.
    """
    h, w = image.shape
    sat = np.cumsum(np.cumsum(image.astype(np.float64), axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)))              # (h+1, w+1): row/col 0 are zeros
    ys = np.linspace(0.0, h, out_wire + 1)           # output-cell edges in input coords
    xs = np.linspace(0.0, w, out_time + 1)
    a = _interp0(sat, ys)                            # sample SAT along wire  -> (out_wire+1, w+1)
    a = _interp0(a.T, xs).T                          # sample SAT along time  -> (out_wire+1, out_time+1)
    box = a[1:, 1:] - a[:-1, 1:] - a[1:, :-1] + a[:-1, :-1]   # per-cell integral (incl-excl)
    area = np.outer(np.diff(ys), np.diff(xs))        # per-cell area in input px
    return (box / area).astype(STORE_DTYPE)


def downscale_to_target(image, target_wire, target_time):
    """Fit ``image`` into (target_wire, target_time) by UNIFORM downscale + center-pad.

    If the crop already fits, it is just center zero-padded (identical to pass 2's
    treatment of small clusters).  If it is larger in either axis, it is shrunk by a
    single scale = min(target_wire/h, target_time/w) < 1 (uniform -> aspect ratio and
    therefore track angles/topology preserved, no distortion), area-averaged to the
    new size, then center-padded to the target.  Returns (padded_image, scale) where
    scale == 1.0 means "fit without downscaling".
    """
    h, w = image.shape
    scale = min(target_wire / h, target_time / w, 1.0)
    if scale >= 1.0:
        return pad_to_target(image, target_wire, target_time), 1.0
    nh = max(1, min(target_wire, int(round(h * scale))))
    nw = max(1, min(target_time, int(round(w * scale))))
    small = area_resize(image, nh, nw)
    return pad_to_target(small, target_wire, target_time), float(scale)


# ------------------------- per-event full image -------------------------------
def build_binary_map(tile_ids, wire_blocks, time_blocks):
    """Binary (wire_blocks, time_blocks) grid from an event's active flat tile ids."""
    bmap = np.zeros((wire_blocks, time_blocks), dtype=np.uint8)
    if tile_ids.size:
        bmap[tile_ids // time_blocks, tile_ids % time_blocks] = 1
    return bmap


def reconstruct_full_image(tiles, tile_ids, wire_blocks, time_blocks, seg_wire, seg_time):
    """Rebuild the (W, T) image from an event's kept tiles (PIXEL_SOURCE='reconstruct')."""
    W, T = wire_blocks * seg_wire, time_blocks * seg_time
    img = np.zeros((W, T), dtype=np.float32)
    for k, ti in enumerate(tile_ids):
        iv, ih = ti // time_blocks, ti % time_blocks
        img[iv * seg_wire:(iv + 1) * seg_wire, ih * seg_time:(ih + 1) * seg_time] = tiles[k]
    return img


# ------------------------- D1 writer (streaming, ragged) ----------------------
# D1 images are variable-size, so we store all pixels concatenated in one 1-D
# dataset ("image_data", wire-major per image) plus per-cluster offsets, rather
# than one HDF5 group per cluster (which would mean ~10^5-10^6 groups).  Metadata
# is parallel 1-D datasets, one row per cluster.  ClusterImages reads it back.
_META_1D = [
    ("source_event_index", np.uint32),
    ("source_file_id", np.uint16),
    ("cluster_id", np.uint32),
    ("n_tiles", np.uint32),
    ("image_width_px", np.uint32),      # wire extent
    ("image_height_px", np.uint32),     # time extent
    ("total_charge", np.float32),
]


class ClusterWriter:
    """Append extracted clusters to D1 in per-file bulk flushes (flat RAM)."""

    def __init__(self, path, attrs):
        self.f = h5py.File(str(path), "w")
        self.f.attrs.update(attrs)
        self.image_data = self.f.create_dataset(
            "image_data", shape=(0,), maxshape=(None,), dtype=STORE_DTYPE,
            chunks=(1 << 16,), compression="gzip", compression_opts=GZIP_LEVEL)
        self.offsets = self.f.create_dataset(
            "image_offsets", shape=(1,), maxshape=(None,), dtype=np.uint64,
            chunks=(4096,), compression="gzip", compression_opts=GZIP_LEVEL)
        self.offsets[0] = 0
        self.meta = {}
        for name, dt in _META_1D:
            self.meta[name] = self.f.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dt,
                chunks=(4096,), compression="gzip", compression_opts=GZIP_LEVEL)
        for name in ("bbox_tile_coords", "bbox_pixel_coords"):
            self.meta[name] = self.f.create_dataset(
                name, shape=(0, 4), maxshape=(None, 4), dtype=np.int32,
                chunks=(4096, 4), compression="gzip", compression_opts=GZIP_LEVEL)
        self.n = 0
        self.total_px = 0

    def append(self, images, rows):
        """images: list of 2-D float32 crops; rows: list of per-cluster meta dicts."""
        b = len(images)
        if b == 0:
            return
        flat = np.concatenate([im.ravel() for im in images]).astype(STORE_DTYPE)
        self.image_data.resize((self.total_px + flat.size,))
        self.image_data[self.total_px:] = flat
        sizes = np.array([im.size for im in images], dtype=np.uint64)
        new_off = self.total_px + np.cumsum(sizes)
        self.offsets.resize((self.n + 1 + b,))
        self.offsets[self.n + 1:] = new_off
        for name, _ in _META_1D:
            d = self.meta[name]
            d.resize((self.n + b,))
            d[self.n:] = [r[name] for r in rows]
        for name in ("bbox_tile_coords", "bbox_pixel_coords"):
            d = self.meta[name]
            d.resize((self.n + b, 4))
            d[self.n:] = np.array([r[name] for r in rows], dtype=np.int32)
        self.n += b
        self.total_px += int(flat.size)

    def close(self, extra_attrs=None):
        if extra_attrs:
            self.f.attrs.update(extra_attrs)
        self.f.attrs["n_clusters"] = self.n
        self.f.close()


class ClusterImages:
    """Reader for a D1/raw cluster file: per-cluster image + metadata access."""

    def __init__(self, path):
        self.h5 = h5py.File(str(path), "r")
        self.image_data = self.h5["image_data"]
        self.offsets = self.h5["image_offsets"][:]
        self.width = self.h5["image_width_px"][:]      # wire extent
        self.height = self.h5["image_height_px"][:]    # time extent
        self.attrs = dict(self.h5.attrs)
        self.source_files = [s.decode() if isinstance(s, bytes) else s
                             for s in self.h5.attrs["source_files"]]

    def __len__(self):
        return int(self.attrs["n_clusters"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.h5.close()

    def image(self, i):
        """The i-th cluster crop as a (wire, time) array."""
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.image_data[lo:hi].reshape(int(self.width[i]), int(self.height[i]))

    def meta(self, i):
        m = {name: self.h5[name][i] for name, _ in _META_1D}
        m["bbox_tile_coords"] = self.h5["bbox_tile_coords"][i]
        m["bbox_pixel_coords"] = self.h5["bbox_pixel_coords"][i]
        m["source_filename"] = self.source_files[int(m["source_file_id"])]
        m["tile_size_used"] = self.attrs["tile_size"]
        return m


# ------------------------------ D2: size histogram ----------------------------
def plot_size_distribution(widths_px, heights_px, n_dropped_big, save_dir):
    """4-panel bounding-box size histogram (guide D2). Returns the stats dict."""
    widths_px = np.asarray(widths_px, dtype=float)
    heights_px = np.asarray(heights_px, dtype=float)
    areas_px = widths_px * heights_px
    pcts = [(50, "green", "--"), (90, "red", "-"), (95, "darkred", ":")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cluster Size Distribution (tiles %s, Q_thresh=%g, N=%d clusters"
                 "%s)" % (SIZE_TAG, 50.0, len(widths_px),
                          "" if not n_dropped_big else ", %d cosmic giants dropped" % n_dropped_big),
                 fontsize=13, fontweight="bold")

    def _hist(ax, data, color, xlabel, title, unit="px"):
        ax.hist(data, bins=60, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        for pct, c, ls in pcts:
            v = np.percentile(data, pct)
            ax.axvline(v, color=c, linestyle=ls, linewidth=2, label="%dth: %.0f %s" % (pct, v, unit))
        ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title); ax.legend(fontsize=8)

    _hist(axes[0, 0], widths_px, "steelblue", "Bounding box width = wire extent (px)", "Width (wire)")
    _hist(axes[0, 1], heights_px, "coral", "Bounding box height = time extent (px)", "Height (time)")
    _hist(axes[1, 0], areas_px, "mediumpurple", "Bounding box area (px^2)", "Area", unit="px^2")

    # Panel 4: width vs height scatter (subsampled) + 90th-pct target rectangle.
    if widths_px.size > SCATTER_MAX:
        sidx = np.random.default_rng(SEED).choice(widths_px.size, SCATTER_MAX, replace=False)
    else:
        sidx = slice(None)
    axes[1, 1].scatter(widths_px[sidx], heights_px[sidx], alpha=0.3, s=8, color="steelblue")
    w90, h90 = np.percentile(widths_px, 90), np.percentile(heights_px, 90)
    axes[1, 1].add_patch(plt.Rectangle((0, 0), w90, h90, fill=False, edgecolor="red",
                                       linewidth=2, label="90th target: %.0f x %.0f" % (w90, h90)))
    axes[1, 1].set_xlabel("Width = wire extent (px)"); axes[1, 1].set_ylabel("Height = time extent (px)")
    axes[1, 1].set_title("Width vs Height (red = proposed padding target)"); axes[1, 1].legend(fontsize=8)

    stats = {
        "w_median": float(np.median(widths_px)), "w_90": float(w90),
        "w_95": float(np.percentile(widths_px, 95)), "w_max": float(widths_px.max()),
        "h_median": float(np.median(heights_px)), "h_90": float(h90),
        "h_95": float(np.percentile(heights_px, 95)), "h_max": float(heights_px.max()),
    }
    stats_text = ("Total clusters: %d\nTile size: %s (wire x time)\nCharge threshold: %g\n"
                  "Width  (wire) - med %.0f, 90th %.0f, max %.0f\n"
                  "Height (time) - med %.0f, 90th %.0f, max %.0f"
                  % (len(widths_px), SIZE_TAG, 50.0,
                     stats["w_median"], stats["w_90"], stats["w_max"],
                     stats["h_median"], stats["h_90"], stats["h_max"]))
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = Path(save_dir) / ("cluster_size_distribution_%s.png" % SIZE_TAG)
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved D2 histogram -> %s" % path)
    return stats


def plot_sample_grid(samples, save_dir):
    """D3: grid of up to N_SAMPLES random extracted clusters, titled with their size."""
    if not samples:
        return
    n = len(samples)
    ncol = 8
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img, title) in zip(axes, samples):
        # display time (axis 1) horizontally, wire (axis 0) vertically
        ax.imshow(img, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(title, fontsize=6)
        ax.axis("off")
    fig.suptitle("Random extracted clusters (%s)  [w=wire px, h=time px]" % SIZE_TAG, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(save_dir) / ("cluster_samples_%s.png" % SIZE_TAG)
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved D3 samples   -> %s" % path)


def plot_clusters_per_event(counts, save_dir):
    """D4: histogram of clusters-per-event (over ALL processed events, zeros included)."""
    counts = np.asarray(counts)
    hi = int(counts.max()) if counts.size else 1
    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=range(0, hi + 2), color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Clusters per event"); plt.ylabel("Count")
    plt.title("Clusters per Event (tiles %s, kept clusters only)" % SIZE_TAG)
    path = Path(save_dir) / ("clusters_per_event_%s.png" % SIZE_TAG)
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved D4 per-event -> %s" % path)


# --------------------------------- PASS 1 -------------------------------------
def run_pass_1():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / RAW_NAME
    if raw_path.exists() and not FORCE:
        raise SystemExit("D1 exists: %s (TILE_CLUSTER_FORCE=1 to overwrite)" % raw_path)

    rng = np.random.default_rng(SEED)
    widths, heights = [], []          # kept-cluster extents (px) -> D2
    per_event_counts = []             # kept clusters per event  -> D4
    samples = []                      # reservoir for D3
    n_events_total = 0
    n_discarded_small = 0
    n_discarded_big = 0
    big_sizes = []                    # (n_tiles) of dropped giants, for D5
    source_files = []
    writer = None
    n_sample_seen = 0

    # D1 is written in size-bounded flushes so RAM stays flat no matter how many
    # events a file has (crops can be several MB each).  Crops are copies, so a
    # flush is safe even while a source file is still open.
    buf_imgs, buf_rows, buf_px = [], [], [0]
    FLUSH_PX = 16_000_000             # ~64 MB of float32 buffered before a flush

    def flush():
        if buf_imgs:
            writer.append(buf_imgs, buf_rows)
            buf_imgs.clear(); buf_rows.clear(); buf_px[0] = 0

    files = list(range(FILE_START, FILE_END + 1))
    for filenum in files:
        tpath = TILE_DIR / tile_name(filenum)
        ipath = IMG_DIR / image_name(filenum)
        if not tpath.exists() or (PIXEL_SOURCE == "original" and not ipath.exists()):
            print("MISS  file %02d (need %s%s)" % (
                filenum, tpath.name, "" if PIXEL_SOURCE != "original" else " and " + ipath.name))
            continue

        with h5py.File(str(tpath), "r") as tf:
            event_index = tf["event_index"][:]
            tile_index = tf["tile_index"][:]
            a = tf.attrs
            wire_blocks, time_blocks = int(a["wire_blocks"]), int(a["time_blocks"])
            n_ev = int(a["n_events"])
            if N_EVENTS is not None:
                n_ev = min(N_EVENTS, n_ev)
            src_name = a.get("source_file", ipath.name)
            src_name = src_name.decode() if isinstance(src_name, bytes) else src_name

            # Lazily-opened pixel source.
            imgf = tiles_ds = None
            cutoff, saturation = FALLBACK_CUTOFF, FALLBACK_SATURATION
            if PIXEL_SOURCE == "original":
                imgf = h5py.File(str(ipath), "r")
                dimg = imgf["image"]
                cutoff = float(dimg.attrs.get("default_cutoff", FALLBACK_CUTOFF))
                saturation = float(dimg.attrs.get("default_saturation", FALLBACK_SATURATION))
            else:
                tiles_ds = tf["tiles"]

            if writer is None:
                writer = ClusterWriter(raw_path, dict(
                    seg_wire=SEG_WIRE, seg_time=SEG_TIME, tile_size=SIZE_TAG,
                    wire_blocks=wire_blocks, time_blocks=time_blocks,
                    image_wire=wire_blocks * SEG_WIRE, image_time=time_blocks * SEG_TIME,
                    plane=PLANE, charge_min=50.0, cutoff=cutoff, saturation=saturation,
                    pad_tiles=PAD_TILES, min_cluster_tiles=MIN_CLUSTER_TILES,
                    max_cluster_tiles=-1 if MAX_CLUSTER_TILES is None else MAX_CLUSTER_TILES,
                    max_span_frac=-1.0 if MAX_SPAN_FRAC is None else MAX_SPAN_FRAC,
                    pixel_source=PIXEL_SOURCE))

            file_id = len(source_files)
            source_files.append(src_name)
            print("FILE %02d  %s  (%d events, pixels=%s)" % (filenum, tpath.name, n_ev, PIXEL_SOURCE))

            # Slice boundaries of each event inside the ascending event_index array.
            los = np.searchsorted(event_index, np.arange(n_ev), "left")
            his = np.searchsorted(event_index, np.arange(n_ev), "right")

            for e in range(n_ev):
                n_events_total += 1
                tids = tile_index[los[e]:his[e]].astype(np.int64)
                if tids.size == 0:
                    per_event_counts.append(0)
                    continue
                bmap = build_binary_map(tids, wire_blocks, time_blocks)
                clusters, disc_small = cluster_binary_map(bmap)
                n_discarded_small += disc_small

                keep = []
                for c in clusters:
                    if is_oversized(c, wire_blocks, time_blocks):
                        n_discarded_big += 1
                        big_sizes.append(c["n_tiles"])
                    else:
                        keep.append(c)
                per_event_counts.append(len(keep))
                if not keep:
                    continue

                # Only now do we pay for pixels (one decompress per non-trivial event).
                if PIXEL_SOURCE == "original":
                    full = apply_cuts(dimg[e], cutoff, saturation)
                else:
                    full = reconstruct_full_image(tiles_ds[los[e]:his[e]], tids,
                                                  wire_blocks, time_blocks, SEG_WIRE, SEG_TIME)

                for c in keep:
                    img, m = extract_cluster_image(full, c["bbox_tile"])
                    widths.append(m["image_width_px"])
                    heights.append(m["image_height_px"])
                    row = {
                        "source_event_index": e, "source_file_id": file_id,
                        "cluster_id": c["cluster_id"], "n_tiles": c["n_tiles"],
                        "image_width_px": m["image_width_px"],
                        "image_height_px": m["image_height_px"],
                        "total_charge": m["total_charge"],
                        "bbox_tile_coords": c["bbox_tile"],
                        "bbox_pixel_coords": m["bbox_pixel_coords"],
                    }
                    buf_imgs.append(img.astype(STORE_DTYPE))
                    buf_rows.append(row)
                    buf_px[0] += int(img.size)

                    # Reservoir sample for D3.
                    title = "%dx%d px, %dt, e%d f%02d" % (
                        m["image_width_px"], m["image_height_px"], c["n_tiles"], e, filenum)
                    n_sample_seen += 1
                    if len(samples) < N_SAMPLES:
                        samples.append((img.copy(), title))
                    else:
                        j = int(rng.integers(n_sample_seen))
                        if j < N_SAMPLES:
                            samples[j] = (img.copy(), title)

                if buf_px[0] >= FLUSH_PX:
                    flush()
                if (e + 1) % 200 == 0 or (e + 1) == n_ev:
                    print("    event %d / %d  (clusters so far: %d)" % (e + 1, n_ev, writer.n + len(buf_imgs)), flush=True)

            flush()                                # release this file's crops before the next
            if imgf is not None:
                imgf.close()

    if writer is None:
        raise SystemExit("No input files found under %s / %s" % (TILE_DIR, IMG_DIR))

    # D2 + stats (kept clusters only).
    if widths:
        stats = plot_size_distribution(widths, heights, n_discarded_big, FIG_DIR)
    else:
        stats = {k: 0.0 for k in ("w_median", "w_90", "w_95", "w_max",
                                  "h_median", "h_90", "h_95", "h_max")}
    tgt_wire = _even_ceil(stats["w_90"]); tgt_time = _even_ceil(stats["h_90"])

    writer.close(extra_attrs=dict(
        source_files=np.array(source_files, dtype="S64"),
        n_events_total=n_events_total,
        n_discarded_small=n_discarded_small, n_discarded_big=n_discarded_big,
        target_wire_90=tgt_wire, target_time_90=tgt_time, **stats))

    # D3, D4.
    plot_sample_grid(samples, FIG_DIR)
    plot_clusters_per_event(per_event_counts, FIG_DIR)

    # D5 summary (console + txt).
    n_clusters = len(widths)
    cpe = np.asarray(per_event_counts)
    big = np.asarray(big_sizes)
    n_at_target = int(np.sum((np.asarray(widths) <= tgt_wire) & (np.asarray(heights) <= tgt_time))) if n_clusters else 0
    lines = [
        "=" * 64, "PASS 1 SUMMARY  (tiles %s, pixels=%s)" % (SIZE_TAG, PIXEL_SOURCE), "=" * 64,
        "Source files processed : %d  (%s)" % (len(source_files), ", ".join(source_files)),
        "Events processed       : %d" % n_events_total,
        "Kept clusters          : %d" % n_clusters,
        "Clusters/event         : mean=%.2f  median=%.0f  max=%d"
        % (cpe.mean() if cpe.size else 0, np.median(cpe) if cpe.size else 0, cpe.max() if cpe.size else 0),
        "Discarded (< %d tiles)   : %d" % (MIN_CLUSTER_TILES, n_discarded_small),
        "Discarded cosmic giants: %d  (cap: >%s tiles or >%s of grid/axis)"
        % (n_discarded_big, MAX_CLUSTER_TILES, MAX_SPAN_FRAC),
        "  dropped-giant n_tiles: median=%s  max=%s"
        % (("%.0f" % np.median(big)) if big.size else "-", ("%d" % big.max()) if big.size else "-"),
        "Charge threshold       : 50 (>=, from nonEmptyImages2)",
        "Width  (wire) px       : median=%.0f  90th=%.0f  95th=%.0f  max=%.0f"
        % (stats["w_median"], stats["w_90"], stats["w_95"], stats["w_max"]),
        "Height (time) px       : median=%.0f  90th=%.0f  95th=%.0f  max=%.0f"
        % (stats["h_median"], stats["h_90"], stats["h_95"], stats["h_max"]),
        "",
        "Proposed padding target (90th pct, rounded to even): %d wire x %d time px" % (tgt_wire, tgt_time),
        "  clusters fitting without crop at target: %d / %d (%.1f%%)"
        % (n_at_target, n_clusters, 100.0 * n_at_target / n_clusters if n_clusters else 0.0),
        "D1 raw clusters -> %s" % (OUT_DIR / RAW_NAME),
        ">> Review D2/D3, then run:  python %s pass2   (or set TILE_CLUSTER_TARGET_WIRE/TIME)" % Path(__file__).name,
        "=" * 64,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    (FIG_DIR / ("cluster_summary_%s.txt" % SIZE_TAG)).write_text(text)


# --------------------------------- PASS 2 -------------------------------------
def run_pass_2():
    raw_path = OUT_DIR / RAW_NAME
    if not raw_path.exists():
        raise SystemExit("D1 not found: %s (run pass 1 first)" % raw_path)

    with ClusterImages(raw_path) as raw:
        n = len(raw)
        widths, heights = raw.width, raw.height       # per-cluster wire/time extents (px)

        # One padded output per requested percentile (default 50th + 75th).  Its target
        # is that percentile of the width/height distribution, rounded up to even.  Check
        # every destination first (so a pre-existing file aborts before any write), then
        # stream each raw crop ONCE and pad it into all outputs.
        plans = {}
        for pct in PAD_PERCENTILES:
            pp = OUT_DIR / padded_name(pct)
            if pp.exists() and not FORCE:
                raise SystemExit("padded file exists: %s (TILE_CLUSTER_FORCE=1 to overwrite)" % pp)
            tw = _even_ceil(np.percentile(widths, pct))
            tt = _even_ceil(np.percentile(heights, pct))
            plans[pct] = dict(path=pp, tw=tw, tt=tt)
            print("PASS 2: %d clusters -> %s  at %d wire x %d time px  (%dth pct)"
                  % (n, pp.name, tw, tt, pct))

        for pct, p in plans.items():
            out = h5py.File(str(p["path"]), "w")
            p["out"] = out
            p["imgs"] = out.create_dataset("images", shape=(n, p["tw"], p["tt"]), dtype=STORE_DTYPE,
                                           chunks=(1, p["tw"], p["tt"]), compression="gzip",
                                           compression_opts=GZIP_LEVEL, shuffle=True)
            for k, v in raw.attrs.items():
                out.attrs[k] = v
            out.attrs["target_wire"] = p["tw"]
            out.attrs["target_time"] = p["tt"]
            out.attrs["target_percentile"] = pct
            p["mg"] = out.create_group("metadata")
            for name, _ in _META_1D:
                p["mg"].create_dataset(name, data=raw.h5[name][:], compression="gzip",
                                       compression_opts=GZIP_LEVEL)
            for name in ("bbox_tile_coords", "bbox_pixel_coords"):
                p["mg"].create_dataset(name, data=raw.h5[name][:], compression="gzip",
                                       compression_opts=GZIP_LEVEL)
            p["was_cropped"] = np.zeros(n, dtype=bool)
            p["zero_frac_sum"] = 0.0

        for i in range(n):
            img = raw.image(i)
            for p in plans.values():
                p["was_cropped"][i] = img.shape[0] > p["tw"] or img.shape[1] > p["tt"]
                padded = pad_to_target(img, p["tw"], p["tt"])
                p["imgs"][i] = padded
                p["zero_frac_sum"] += float(np.mean(padded == 0))
            if (i + 1) % 5000 == 0 or (i + 1) == n:
                print("    padded %d / %d" % (i + 1, n), flush=True)

        print("\n" + "=" * 64)
        print("PASS 2 COMPLETE  (%d percentile file(s))" % len(plans))
        for pct, p in plans.items():
            p["mg"].create_dataset("was_cropped", data=p["was_cropped"],
                                   compression="gzip", compression_opts=GZIP_LEVEL)
            n_crop = int(p["was_cropped"].sum())
            print("%dth pct -> %s   (%d wire x %d time px)"
                  % (pct, p["path"].name, p["tw"], p["tt"]))
            print("   cropped (exceeded target): %d (%.1f%%)  [aim < 10-15%%]"
                  % (n_crop, 100.0 * n_crop / n if n else 0.0))
            print("   mean zero (padding) fraction: %.1f%%  [above ~90%% means target too large]"
                  % (100.0 * p["zero_frac_sum"] / n if n else 0.0))
            p["out"].close()
        print("=" * 64)


# --------------------------------- PASS 3 -------------------------------------
def run_pass_3():
    """Alternate training files: DOWNSCALE oversized clusters (instead of cropping) and
    drop <=(PASS3_MIN_TILES-1)-tile blips.  Writes clusters_padded_downscaled_<pct>_<size>.h5,
    one per percentile.  Targets are the SAME as pass 2 (that percentile of the full raw
    extent distribution), so downscaled_75 (288x176) is a controlled crop-vs-downscale
    A/B against clusters_padded_75.  Pass 1/2 untouched."""
    raw_path = OUT_DIR / RAW_NAME
    if not raw_path.exists():
        raise SystemExit("D1 not found: %s (run pass 1 first)" % raw_path)

    with ClusterImages(raw_path) as raw:
        n = len(raw)
        widths, heights = raw.width, raw.height            # per-cluster wire/time extents (px)
        n_tiles = raw.h5["n_tiles"][:]

        # (b) drop small blips: keep only clusters with >= PASS3_MIN_TILES tiles.
        keep_idx = np.nonzero(n_tiles >= PASS3_MIN_TILES)[0]
        n_keep = int(keep_idx.size)
        n_drop = n - n_keep
        print("PASS 3: dropping %d / %d clusters with <= %d tiles (blips); keeping %d"
              % (n_drop, n, PASS3_MIN_TILES - 1, n_keep))
        if n_keep == 0:
            raise SystemExit("PASS 3: nothing left after the <=%d-tile cut" % (PASS3_MIN_TILES - 1))

        # Targets identical to pass 2 (percentiles of the FULL raw distribution), so the
        # only per-cluster difference vs clusters_padded_<pct> is downscale-instead-of-crop.
        plans = {}
        for pct in PAD_PERCENTILES:
            pp = OUT_DIR / padded_downscaled_name(pct)
            if pp.exists() and not FORCE:
                raise SystemExit("downscaled file exists: %s (TILE_CLUSTER_FORCE=1 to overwrite)" % pp)
            tw = _even_ceil(np.percentile(widths, pct))
            tt = _even_ceil(np.percentile(heights, pct))
            plans[pct] = dict(path=pp, tw=tw, tt=tt)
            print("PASS 3: %d kept clusters -> %s  at %d wire x %d time px  (%dth pct, DOWNSCALE)"
                  % (n_keep, pp.name, tw, tt, pct))

        for pct, p in plans.items():
            out = h5py.File(str(p["path"]), "w")
            p["out"] = out
            p["imgs"] = out.create_dataset("images", shape=(n_keep, p["tw"], p["tt"]),
                                           dtype=STORE_DTYPE, chunks=(1, p["tw"], p["tt"]),
                                           compression="gzip", compression_opts=GZIP_LEVEL, shuffle=True)
            for k, v in raw.attrs.items():
                out.attrs[k] = v
            out.attrs["target_wire"] = p["tw"]
            out.attrs["target_time"] = p["tt"]
            out.attrs["target_percentile"] = pct
            out.attrs["oversize_policy"] = "downscale"            # vs pass 2's "crop"
            out.attrs["pass3_min_tiles"] = PASS3_MIN_TILES
            out.attrs["n_clusters"] = n_keep
            out.attrs["n_dropped_blips"] = n_drop
            # metadata for the KEPT clusters only, plus raw_index so truth labels (indexed
            # by raw-cluster order) can be mapped:  is_nu_downscaled = is_nu[raw_index].
            mg = out.create_group("metadata")
            p["mg"] = mg
            for name, _ in _META_1D:
                mg.create_dataset(name, data=raw.h5[name][:][keep_idx],
                                  compression="gzip", compression_opts=GZIP_LEVEL)
            for name in ("bbox_tile_coords", "bbox_pixel_coords"):
                mg.create_dataset(name, data=raw.h5[name][:][keep_idx],
                                  compression="gzip", compression_opts=GZIP_LEVEL)
            mg.create_dataset("raw_index", data=keep_idx.astype(np.uint32),
                              compression="gzip", compression_opts=GZIP_LEVEL)
            p["was_downscaled"] = np.zeros(n_keep, dtype=bool)
            p["downscale_factor"] = np.ones(n_keep, dtype=np.float32)
            p["zero_frac_sum"] = 0.0

        for out_row, i in enumerate(keep_idx):
            img = raw.image(int(i))
            for p in plans.values():
                proc, factor = downscale_to_target(img, p["tw"], p["tt"])
                p["imgs"][out_row] = proc
                p["downscale_factor"][out_row] = factor
                p["was_downscaled"][out_row] = factor < 1.0
                p["zero_frac_sum"] += float(np.mean(proc == 0))
            if (out_row + 1) % 5000 == 0 or (out_row + 1) == n_keep:
                print("    processed %d / %d" % (out_row + 1, n_keep), flush=True)

        print("\n" + "=" * 64)
        print("PASS 3 COMPLETE  (%d percentile file(s); oversize policy = DOWNSCALE)" % len(plans))
        for pct, p in plans.items():
            p["mg"].create_dataset("was_downscaled", data=p["was_downscaled"],
                                   compression="gzip", compression_opts=GZIP_LEVEL)
            p["mg"].create_dataset("downscale_factor", data=p["downscale_factor"],
                                   compression="gzip", compression_opts=GZIP_LEVEL)
            n_ds = int(p["was_downscaled"].sum())
            fac = p["downscale_factor"][p["was_downscaled"]]
            print("%dth pct -> %s   (%d wire x %d time px)" % (pct, p["path"].name, p["tw"], p["tt"]))
            print("   downscaled (exceeded target): %d (%.1f%%)  [pass 2 would CROP these instead]"
                  % (n_ds, 100.0 * n_ds / n_keep if n_keep else 0.0))
            print("   downscale factor on those   : median=%s  min=%s  (1.0 = untouched)"
                  % (("%.3f" % np.median(fac)) if fac.size else "-",
                     ("%.3f" % fac.min()) if fac.size else "-"))
            print("   mean zero (padding) fraction: %.1f%%" % (100.0 * p["zero_frac_sum"] / n_keep if n_keep else 0.0))
            p["out"].close()
        print("dropped %d blip clusters (<= %d tiles)" % (n_drop, PASS3_MIN_TILES - 1))
        print("=" * 64)


# --------------------------------- self-test ----------------------------------
def _self_test():
    """Logic checks on a synthetic binary map -- no data files needed."""
    wb, tb = 20, 10
    bmap = np.zeros((wb, tb), dtype=np.uint8)
    bmap[2:5, 2:4] = 1                     # a 3x2 = 6-tile cluster
    bmap[2, 6] = 1                         # a 1-tile noise hit (dropped by MIN_CLUSTER_TILES)
    bmap[8, 0:tb] = 1                      # a full-time-spanning "cosmic" (10 tiles wide in time)
    clusters, small = cluster_binary_map(bmap, min_tiles=3)
    assert small == 1, "expected 1 sub-min cluster dropped, got %d" % small
    labels = {c["n_tiles"] for c in clusters}
    assert 6 in labels, "6-tile cluster missing: %s" % labels
    big = [c for c in clusters if is_oversized(c, wb, tb)]
    assert len(big) == 1 and big[0]["n_tiles"] == tb, "span cap should flag the full-time cluster"
    # extraction geometry
    full = np.zeros((wb * SEG_WIRE, tb * SEG_TIME), dtype=np.float32)
    c6 = next(c for c in clusters if c["n_tiles"] == 6)
    img, m = extract_cluster_image(full, c6["bbox_tile"], pad_tiles=1)
    exp_w = (c6["bbox_tile"][1] - c6["bbox_tile"][0] + 1 + 2) * SEG_WIRE
    exp_h = (c6["bbox_tile"][3] - c6["bbox_tile"][2] + 1 + 2) * SEG_TIME
    assert img.shape == (exp_w, exp_h) == (m["image_width_px"], m["image_height_px"]), img.shape
    # padding round-trips a small image, centered
    p = pad_to_target(np.ones((exp_w, exp_h), np.float32), _even_ceil(exp_w) + 4, _even_ceil(exp_h) + 4)
    assert p.shape == (_even_ceil(exp_w) + 4, _even_ceil(exp_h) + 4) and p.sum() == exp_w * exp_h
    print("[ok] clustering, cosmic caps, extraction geometry, and padding all pass")

    # pass-3 area downscale: exactness (flat patch), topology (diagonal survives), fit + pad
    flat = np.full((100, 60), 7.0, np.float32)
    r = area_resize(flat, 20, 12)
    assert r.shape == (20, 12) and np.allclose(r, 7.0), "area_resize must be mean-preserving on flat input"
    diag = np.zeros((64, 64), np.float32); np.fill_diagonal(diag, 100.0)
    rd = area_resize(diag, 16, 16)
    assert (np.diag(rd) > 0).all(), "diagonal track must survive downscale (topology preserved)"
    assert rd.max() <= 100.0 + 1e-3, "area average cannot exceed input max"
    big = np.ones((400, 300), np.float32)                 # bigger than target in both axes
    out, sc = downscale_to_target(big, 288, 176)
    assert out.shape == (288, 176) and 0.0 < sc < 1.0, "oversized -> uniform downscale + pad"
    assert abs(sc - min(288 / 400, 176 / 300)) < 1e-6, "scale must be the min aspect-preserving ratio"
    small = np.ones((50, 40), np.float32)                 # already fits -> just pad, scale 1.0
    out2, sc2 = downscale_to_target(small, 288, 176)
    assert out2.shape == (288, 176) and sc2 == 1.0 and out2.sum() == 50 * 40, "fitting crop is only padded"
    print("[ok] pass-3 area downscale: exact on flats, topology-preserving, correct fit/scale")
    print("SELF-TEST PASSED")


def main():
    if "--self-test" in sys.argv:
        _self_test(); return
    if len(sys.argv) > 1 and sys.argv[1] == "pass2":
        run_pass_2()
    elif len(sys.argv) > 1 and sys.argv[1] == "pass3":
        run_pass_3()
    else:
        print("Tile map dir : %s" % TILE_DIR)
        print("Pixel source : %s%s" % (PIXEL_SOURCE, "  (%s)" % IMG_DIR if PIXEL_SOURCE == "original" else ""))
        print("Output dir   : %s   Figures: %s" % (OUT_DIR, FIG_DIR))
        print("Size %s  files %02d..%02d  min_tiles=%d  cap=(>%s tiles | >%s span)  pad=%d\n"
              % (SIZE_TAG, FILE_START, FILE_END, MIN_CLUSTER_TILES,
                 MAX_CLUSTER_TILES, MAX_SPAN_FRAC, PAD_TILES))
        run_pass_1()


if __name__ == "__main__":
    main()
