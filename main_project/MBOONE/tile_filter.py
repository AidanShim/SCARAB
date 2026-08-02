"""Charge-thresholded tile loader for autoencoder training.

Pairs the processed plane-2 images (plane_2.py) with the per-tile manifest
(tile_stats.py) to yield ONLY tiles whose summed cut charge clears a threshold,
so the autoencoder is not trained on mostly-empty images.

The threshold lives here, not in the stored data: the manifest holds raw
per-tile charge, so retuning CHARGE_MIN costs nothing and reprocesses nothing.

Note on the threshold scale: preprocessing zeroes every pixel below the cutoff
(10 ADC), so any surviving pixel contributes >= 10 charge. Hence
``charge > 0`` is exactly equivalent to "at least one surviving pixel", and
``charge >= 50`` is the charge analogue of the paper's rule that a cluster needs
>= 5 hit instances to count as a track.

Typical use:

    from tile_filter import FilteredTiles

    data = FilteredTiles(seg_wire=64, seg_time=32, charge_min=50)
    print(len(data), "surviving tiles of", data.n_total_tiles)

    for X, Y in data.batches(batch_size=256, shuffle=True):
        model.train_on_batch(X, Y)
"""

import os
from pathlib import Path

import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROC_DIR = REPO_ROOT / "data" / "ubopendata" / "processed_npy"

DEFAULT_CHARGE_MIN = 50.0   # = 5 x the 10-ADC pixel cutoff


def apply_cuts(img, cutoff, saturation):
    """Zero sub-threshold pixels, clip saturated ones. Returns a float32 copy."""
    out = np.asarray(img, dtype=np.float32).copy()
    out[out < cutoff] = 0
    out[out > saturation] = saturation
    return out


def split_into_segments(img, seg_wire, seg_time):
    """Tile a (W, T) image into (n_tiles, seg_wire, seg_time, 1) crops.

    Same ordering as mod_data_loader.split_into_segments; tile_stats.py asserts
    its manifest indices agree with this.
    """
    v_split = img.shape[0] // seg_wire
    h_split = img.shape[1] // seg_time
    X = np.array(np.split(img, h_split, axis=1))
    X = np.array(np.split(X, v_split, axis=1))
    return np.reshape(X, (-1, seg_wire, seg_time, 1))


class FilteredTiles:
    """Index of tiles passing a charge threshold, with batched access.

    Builds a compact global index of surviving (file, event, tile) triples from
    the manifest, then reads image data lazily one event at a time.
    """

    def __init__(self, seg_wire, seg_time, charge_min=DEFAULT_CHARGE_MIN,
                 proc_dir=None, files=range(18)):
        self.seg_wire, self.seg_time = seg_wire, seg_time
        self.charge_min = float(charge_min)
        self.proc_dir = Path(proc_dir) if proc_dir else DEFAULT_PROC_DIR
        self.key = "%dx%d" % (seg_wire, seg_time)

        fi_l, ev_l, ti_l = [], [], []
        self.n_total_tiles = 0
        self.files = []
        for f in files:
            stats = self.proc_dir / ("bnb_WithWire_%02d_tilestats.h5" % f)
            if not stats.exists():
                continue
            with h5py.File(str(stats), "r") as h:
                if self.key not in h:
                    raise KeyError("%s has no group %s" % (stats.name, self.key))
                charge = h[self.key]["charge"][:]        # (n_events, n_tiles)
            self.n_total_tiles += charge.size
            ev, ti = np.nonzero(charge >= self.charge_min)
            fi_l.append(np.full(ev.size, f, dtype=np.uint8))
            ev_l.append(ev.astype(np.uint32))
            ti_l.append(ti.astype(np.uint16))
            self.files.append(f)
            del charge

        if not self.files:
            raise FileNotFoundError(
                "No tile-stats files found in %s. Run tile_stats.py first." % self.proc_dir)

        self.file_idx = np.concatenate(fi_l)
        self.event_idx = np.concatenate(ev_l)
        self.tile_idx = np.concatenate(ti_l)
        self._images = {}

    def __len__(self):
        return self.file_idx.size

    @property
    def kept_fraction(self):
        return len(self) / float(self.n_total_tiles) if self.n_total_tiles else 0.0

    def close(self):
        for h in self._images.values():
            h.close()
        self._images = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _image_handle(self, f):
        if f not in self._images:
            p = self.proc_dir / ("bnb_WithWire_%02d_plane2_images.h5" % f)
            self._images[f] = h5py.File(str(p), "r")
        return self._images[f]

    def _event_tiles(self, f, ev, tiles):
        """Cut + tile one event, returning only the requested tile indices."""
        d = self._image_handle(f)["image"]
        cutoff = float(d.attrs.get("default_cutoff", 10))
        saturation = float(d.attrs.get("default_saturation", 100))
        img = apply_cuts(d[ev], cutoff, saturation)
        return split_into_segments(img, self.seg_wire, self.seg_time)[tiles]

    def batches(self, batch_size, autoencoder=True, shuffle=False, seed=0, limit=None):
        """Yield batches of surviving tiles, decompressing each event once.

        Yields (X, X) when autoencoder=True, else X, with X of shape
        (batch_size, seg_wire, seg_time, 1). Events are visited in order (or
        shuffled); all surviving tiles of an event are emitted together.
        """
        # group surviving tiles by (file, event) so each event is read once
        order = np.lexsort((self.event_idx, self.file_idx))
        f_s, e_s, t_s = self.file_idx[order], self.event_idx[order], self.tile_idx[order]
        bounds = np.flatnonzero(np.diff(f_s.astype(np.int64) * (2 ** 32) + e_s)) + 1
        groups = np.split(np.arange(f_s.size), bounds)
        if shuffle:
            np.random.default_rng(seed).shuffle(groups)
        if limit is not None:
            groups = groups[:limit]

        buf = []
        for g in groups:
            tiles = self._event_tiles(int(f_s[g[0]]), int(e_s[g[0]]), t_s[g])
            for seg in tiles:
                buf.append(seg)
                if len(buf) == batch_size:
                    X = np.stack(buf); buf = []
                    yield (X, X) if autoencoder else X
        if buf:
            X = np.stack(buf)
            yield (X, X) if autoencoder else X


def threshold_scan(seg_wire, seg_time, thresholds, proc_dir=None, files=range(18)):
    """Fraction of tiles surviving each candidate charge threshold."""
    proc_dir = Path(proc_dir) if proc_dir else DEFAULT_PROC_DIR
    key = "%dx%d" % (seg_wire, seg_time)
    total = 0
    kept = np.zeros(len(thresholds), dtype=np.int64)
    for f in files:
        stats = proc_dir / ("bnb_WithWire_%02d_tilestats.h5" % f)
        if not stats.exists():
            continue
        with h5py.File(str(stats), "r") as h:
            charge = h[key]["charge"][:]
        total += charge.size
        flat = charge.reshape(-1)
        for i, thr in enumerate(thresholds):
            kept[i] += int(np.count_nonzero(flat >= thr))
        del charge, flat
    return total, kept
