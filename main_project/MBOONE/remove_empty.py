# Materialize non-empty tile sets for autoencoder training.
#
# For each of the 18 processed plane-2 image files and each of the 3 tile sizes
# (864x64, 64x32, 18x16) this writes ONE file of only the tiles that survive the
# emptiness cut -- 3 x 18 = 54 files in total. A tile is kept when its summed
# charge (AFTER the 10-ADC cutoff and 100 saturation clip) is > 0; i.e. we drop
# every tile that is entirely zero after cuts.
#
# WHY MATERIALIZE (vs tile_filter.FilteredTiles, which filters lazily on load):
#   - hands a dense, training-ready array straight to a training.py (--input),
#   - and, crucially, is fully RECONSTRUCTABLE: each kept tile is stored with the
#     (event, tile) it came from, so any event can be rebuilt into the original
#     3456x640 image with the empty tiles removed (left as zeros).
#
# LOSSLESS BY CONSTRUCTION: the three tile sizes each tile the whole image with
# no gaps or overlap, and every dropped tile is all-zero, so reassembling the
# kept tiles reproduces the cut image EXACTLY. reconstruct_event() therefore
# returns the preprocessed image with nothing lost but the empty space.
#
# Output files are written to main_project/MBOONE/nonEmptyImages/ (override with
# REMOVE_EMPTY_OUT).
#
# WHAT EACH OUTPUT FILE HOLDS  (bnb_WithWire_XX_plane2_<W>x<T>_nonempty.h5):
#   tiles        (n_kept, seg_wire, seg_time) float32  -- the CUT pixel data
#   event_index  (n_kept,)                    uint32   -- source event of each tile
#   tile_index   (n_kept,)                    uint16   -- tile's slot in split order
#   attrs: seg_wire, seg_time, wire_blocks, time_blocks, n_events, cutoff,
#          saturation, keep_rule, plane, source_file
#   event_index/tile_index are in ascending (event, tile) order, so a single
#   event's tiles occupy a contiguous run (found with searchsorted).
#
# The tile numbering matches mod_data_loader.split_into_segments and the
# tile_stats.py catalog: flat index = wire_block * time_blocks + time_block.
#
# RUN (from the SCARAB repo root) with the ubopendata env active:
#   conda run --no-capture-output -n ubopendata python main_project/MBOONE/remove_empty.py
# A different threshold + destination (e.g. keep charge >= 50 into nonEmptyImages2):
#   REMOVE_EMPTY_CHARGE_MIN=50 REMOVE_EMPTY_OUT=main_project/MBOONE/nonEmptyImages2 \
#     conda run --no-capture-output -n ubopendata python main_project/MBOONE/remove_empty.py
# Subset / test:
#   MOD_DATA_FILE_START=0 MOD_DATA_FILE_END=0 REMOVE_EMPTY_N_EVENTS=8 ... python .../remove_empty.py
#   REMOVE_EMPTY_FORCE=1 ... python .../remove_empty.py     # overwrite existing outputs

import os
import gc
from pathlib import Path

import numpy as np
import h5py

# ----------------------------- Configuration -----------------------------
PLANE = 2
SEGMENT_SIZES = [(864, 64), (64, 32), (18, 16)]

# Fallback cuts; the real values are read from each image file's attrs.
FALLBACK_CUTOFF = 10.0
FALLBACK_SATURATION = 100.0

STORE_DTYPE = np.float32
GZIP_LEVEL = 4

FILE_START = int(os.environ.get("MOD_DATA_FILE_START", 0))
FILE_END = int(os.environ.get("MOD_DATA_FILE_END", 17))
FORCE = bool(int(os.environ.get("REMOVE_EMPTY_FORCE", "0")))

# Emptiness threshold: a tile is KEPT when its summed charge (after cuts) clears
# this value. 0 (default) means "keep any tile with charge > 0", i.e. drop only
# all-zero tiles. Set e.g. 50 to drop tiles with charge < 50. Because the 10-ADC
# pixel cutoff means no tile can have charge in (0, 10), a threshold of 0 and any
# threshold <= 10 keep exactly the same tiles.
CHARGE_MIN = float(os.environ.get("REMOVE_EMPTY_CHARGE_MIN", "0"))

N_EVENTS = None
if os.environ.get("REMOVE_EMPTY_N_EVENTS"):
    N_EVENTS = int(os.environ["REMOVE_EMPTY_N_EVENTS"])

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROC_DIR = Path(os.environ.get("MOD_DATA_OUT",
                               REPO_ROOT / "data" / "ubopendata" / "processed_npy"))
# The 54 non-empty tile files land in main_project/MBOONE/nonEmptyImages/.
OUT_DIR = Path(os.environ.get("REMOVE_EMPTY_OUT", SCRIPT_DIR / "nonEmptyImages"))
# -------------------------------------------------------------------------


def apply_cuts(img, cutoff, saturation):
    """Zero sub-threshold pixels, clip saturated ones. Returns a float32 copy."""
    out = np.asarray(img, dtype=np.float32).copy()
    out[out < cutoff] = 0
    out[out > saturation] = saturation
    return out


def split_into_segments(img, seg_wire, seg_time):
    """Tile a (W, T) image into (n_tiles, seg_wire, seg_time) crops.

    Flat order matches mod_data_loader.split_into_segments and the tile_stats
    catalog: wire-block major, time-block fastest.
    """
    v_split = img.shape[0] // seg_wire
    h_split = img.shape[1] // seg_time
    X = np.array(np.split(img, h_split, axis=1))   # (h_split, W, seg_time)
    X = np.array(np.split(X, v_split, axis=1))     # (v_split, h_split, seg_wire, seg_time)
    return np.reshape(X, (-1, seg_wire, seg_time))


def out_name(filenum, seg_wire, seg_time):
    return "bnb_WithWire_%02d_plane%d_%dx%d_nonempty.h5" % (filenum, PLANE, seg_wire, seg_time)


def process_file(filenum):
    """Write the 3 non-empty tile files for one source file. Returns (written, skipped)."""
    img_path = PROC_DIR / ("bnb_WithWire_%02d_plane%d_images.h5" % (filenum, PLANE))
    cat_path = PROC_DIR / ("bnb_WithWire_%02d_tilestats.h5" % filenum)
    if not img_path.exists() or not cat_path.exists():
        print("MISS  file %02d (need both %s and %s)" % (filenum, img_path.name, cat_path.name))
        return [], []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []

    with h5py.File(str(img_path), "r") as imgf, h5py.File(str(cat_path), "r") as catf:
        d = imgf["image"]
        N, W, T = d.shape
        n_total = N if N_EVENTS is None else min(N_EVENTS, N)
        cutoff = float(d.attrs.get("default_cutoff", FALLBACK_CUTOFF))
        saturation = float(d.attrs.get("default_saturation", FALLBACK_SATURATION))

        # ---- per size: decide keeps from the catalog, preallocate outputs ----
        plan = {}   # size -> dict(out, tiles, ptr, keep, nv, nh)
        for sw, st in SEGMENT_SIZES:
            op = OUT_DIR / out_name(filenum, sw, st)
            if op.exists() and not FORCE:
                print("SKIP  %s (exists; REMOVE_EMPTY_FORCE=1 to overwrite)" % op.name)
                skipped.append((filenum, sw, st))
                continue

            charge = catf["%dx%d" % (sw, st)]["charge"][:n_total]     # (n_total, ntiles)
            keep = charge >= CHARGE_MIN if CHARGE_MIN > 0 else charge > 0.0
            ev_idx, tile_idx = np.nonzero(keep)                        # ascending (event, tile)
            n_kept = int(ev_idx.size)
            nv, nh = W // sw, T // st

            out = h5py.File(str(op), "w")
            out.create_dataset("tiles", shape=(n_kept, sw, st), dtype=STORE_DTYPE,
                               chunks=(1, sw, st), compression="gzip",
                               compression_opts=GZIP_LEVEL, shuffle=True)
            out.create_dataset("event_index", data=ev_idx.astype(np.uint32),
                               compression="gzip", compression_opts=GZIP_LEVEL)
            out.create_dataset("tile_index", data=tile_idx.astype(np.uint16),
                               compression="gzip", compression_opts=GZIP_LEVEL)
            keep_rule = ("charge >= %g (after cuts)" % CHARGE_MIN if CHARGE_MIN > 0
                         else "charge > 0 (after cuts)")
            out.attrs.update(seg_wire=sw, seg_time=st, wire_blocks=nv, time_blocks=nh,
                             n_events=n_total, image_wire=W, image_time=T,
                             cutoff=cutoff, saturation=saturation, plane=PLANE,
                             charge_min=CHARGE_MIN, keep_rule=keep_rule,
                             source_file=img_path.name)
            plan[(sw, st)] = dict(out=out, tiles=out["tiles"], ptr=0, keep=keep, nv=nv, nh=nh)
            print("  %s: keeping %d / %d tiles (%.1f%%)"
                  % (op.name, n_kept, keep.size, 100.0 * n_kept / keep.size))

        if not plan:                       # all three already existed
            return written, skipped

        # ---- one pass over the images, filling every size at once ----
        for e in range(n_total):
            if not any(p["keep"][e].any() for p in plan.values()):
                continue                   # empty event: nothing kept at any size
            cutimg = apply_cuts(d[e], cutoff, saturation)
            for size, p in plan.items():
                sel = np.nonzero(p["keep"][e])[0]
                if sel.size == 0:
                    continue
                tiles = split_into_segments(cutimg, *size)[sel]
                p["tiles"][p["ptr"]:p["ptr"] + sel.size] = tiles
                p["ptr"] += sel.size
            if (e + 1) % 200 == 0 or (e + 1) == n_total:
                print("    processed %d / %d events" % (e + 1, n_total), flush=True)

        for size, p in plan.items():
            assert p["ptr"] == p["tiles"].shape[0], "row count mismatch for %s" % (size,)
            p["out"].close()
            written.append((filenum,) + size)
        gc.collect()

    return written, skipped


def main():
    keep_desc = "charge >= %g" % CHARGE_MIN if CHARGE_MIN > 0 else "charge > 0"
    print("Image/catalog dir: %s" % PROC_DIR)
    print("Output dir       : %s" % OUT_DIR)
    print("Keep rule        : %s (after cuts)" % keep_desc)
    print("Files %02d..%02d  (force=%s, n_events=%s)\n" % (FILE_START, FILE_END, FORCE, N_EVENTS))

    all_written, all_skipped, missing = [], [], []
    for filenum in range(FILE_START, FILE_END + 1):
        w, s = process_file(filenum)
        if not w and not s:
            missing.append(filenum)
        all_written += w
        all_skipped += s
        print()

    print("=" * 60)
    print("Done. wrote %d files, skipped %d, missing sources %s"
          % (len(all_written), len(all_skipped), missing))


# --------------------------- Reconstruction ---------------------------------
class NonEmptyTiles:
    """Reader for a non-empty tile file: training access + image reconstruction."""

    def __init__(self, path):
        self.h5 = h5py.File(str(path), "r")
        self.tiles = self.h5["tiles"]
        self.event_index = self.h5["event_index"][:]     # small; keep in RAM for searchsorted
        self.tile_index = self.h5["tile_index"][:]
        a = self.tiles.attrs if "seg_wire" in self.tiles.attrs else self.h5.attrs
        self.sw = int(a["seg_wire"]); self.st = int(a["seg_time"])
        self.nv = int(a["wire_blocks"]); self.nh = int(a["time_blocks"])
        self.W = int(a["image_wire"]); self.T = int(a["image_time"])
        self.n_events = int(a["n_events"])

    def __len__(self):
        return self.tiles.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.h5.close()

    def reconstruct_event(self, event_index):
        """Rebuild one event's (W, T) image from its kept tiles; dropped (empty)
        tiles stay zero. Lossless vs the cut source image."""
        lo = int(np.searchsorted(self.event_index, event_index, "left"))
        hi = int(np.searchsorted(self.event_index, event_index, "right"))
        img = np.zeros((self.W, self.T), dtype=np.float32)
        for r in range(lo, hi):
            ti = int(self.tile_index[r])
            iv, ih = ti // self.nh, ti % self.nh
            img[iv * self.sw:(iv + 1) * self.sw, ih * self.st:(ih + 1) * self.st] = self.tiles[r]
        return img

    def batches(self, batch_size, autoencoder=True, shuffle=False, seed=0):
        """Yield training batches of shape (b, seg_wire, seg_time, 1).

        Reads sequentially (fast) unless shuffle=True, which permutes tile order.
        """
        n = len(self)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for i in range(0, n, batch_size):
            idx = np.sort(order[i:i + batch_size])       # sorted -> valid h5py fancy index
            X = self.tiles[idx][..., None].astype(np.float32)
            yield (X, X) if autoencoder else X


if __name__ == "__main__":
    main()
