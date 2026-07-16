"""Lazy loader for the compressed plane-2 image file written by mod_data.py.

mod_data.py stores ONE array: downsampled, NO-CUT plane-2 images
(shape (N, 3456, 640), float16, gzip-chunked). This module applies the pixel
cuts and tiles images into segments on demand, so you never hold the whole
(uncompressed) dataset in memory.

Typical use for training:

    from mod_data_loader import WireImages

    data = WireImages("data/ubopendata/processed_npy/bnb_WithWire_00_plane2_images.h5")
    print(len(data), "events")

    # all 864x64 segments, one event at a time (memory-safe):
    for seg in data.iter_segments(864, 64):     # seg: (864, 64, 1) float32
        train_step(seg)

    # or grab every segment of one event as a batch:
    X = data.segments(event_index=0, seg_wire=864, seg_time=64)   # (40, 864, 64, 1)
"""

import os
import random

import numpy as np
import h5py
from numpy.lib.format import open_memmap


def apply_cuts(img, cutoff, saturation):
    """Zero sub-threshold pixels, clip saturated ones. Returns float32 copy."""
    out = np.asarray(img, dtype=np.float32).copy()
    out[out < cutoff] = 0
    out[out > saturation] = saturation
    return out


def split_into_segments(img, seg_wire, seg_time):
    """Tile a (W, T) image into (n_segments, seg_wire, seg_time, 1) contiguous crops."""
    v_split = img.shape[0] // seg_wire
    h_split = img.shape[1] // seg_time
    X = np.array(np.split(img, h_split, axis=1))   # (h_split, W, seg_time)
    X = np.array(np.split(X, v_split, axis=1))     # (v_split, h_split, seg_wire, seg_time)
    return np.reshape(X, (-1, seg_wire, seg_time, 1))


class WireImages:
    """Lazy accessor over the compressed no-cut plane-2 image dataset."""

    def __init__(self, path):
        self.h5 = h5py.File(path, "r")
        self.ds = self.h5["image"]
        self.cutoff = float(self.ds.attrs.get("default_cutoff", 10))
        self.saturation = float(self.ds.attrs.get("default_saturation", 100))

    def __len__(self):
        return self.ds.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.h5.close()

    def full_image(self, event_index, cut=False):
        """Return one (W, T) float32 image; raw by default, or with cuts applied."""
        img = np.asarray(self.ds[event_index], dtype=np.float32)
        return apply_cuts(img, self.cutoff, self.saturation) if cut else img

    def segments(self, event_index, seg_wire, seg_time, cutoff=None, saturation=None):
        """Return (n_seg, seg_wire, seg_time, 1) cut+tiled segments for one event."""
        c = self.cutoff if cutoff is None else cutoff
        s = self.saturation if saturation is None else saturation
        img = apply_cuts(self.ds[event_index], c, s)
        return split_into_segments(img, seg_wire, seg_time)

    def iter_segments(self, seg_wire, seg_time, cutoff=None, saturation=None):
        """Yield individual (seg_wire, seg_time, 1) segments across all events."""
        for i in range(len(self)):
            for seg in self.segments(i, seg_wire, seg_time, cutoff, saturation):
                yield seg

    def segments_per_event(self, seg_wire, seg_time):
        """Number of tiles one event produces at this segment size (e.g. 40 for 864x64)."""
        return (self.ds.shape[1] // seg_wire) * (self.ds.shape[2] // seg_time)

    def num_segments(self, seg_wire, seg_time, event_indices=None):
        """Total segments (use // batch_size for Keras steps_per_epoch)."""
        n = len(self) if event_indices is None else len(event_indices)
        return n * self.segments_per_event(seg_wire, seg_time)

    def batches(self, seg_wire, seg_time, batch_size, event_indices=None,
                autoencoder=True, shuffle=False, cutoff=None, saturation=None):
        """Yield batches of segments, tiling one event at a time (memory-safe).

        Each item is (X, X) when autoencoder=True (input == reconstruction target),
        else just X. X has shape (batch_size, seg_wire, seg_time, 1). Pass
        event_indices to restrict to a train/val/test split; shuffle=True reshuffles
        event order each call (i.e. each epoch if you re-call the generator).
        """
        order = list(range(len(self)) if event_indices is None else event_indices)
        if shuffle:
            random.shuffle(order)
        buf = []
        for i in order:
            for seg in self.segments(i, seg_wire, seg_time, cutoff, saturation):
                buf.append(seg)
                if len(buf) == batch_size:
                    X = np.stack(buf); buf = []
                    yield (X, X) if autoencoder else X
        if buf:  # last partial batch
            X = np.stack(buf)
            yield (X, X) if autoencoder else X

    def export_npy(self, out_path, seg_wire, seg_time, event_indices=None):
        """Write all cut+tiled segments to a dense .npy for training.py (memory-safe).

        Output shape is (n_events * segments_per_event, seg_wire, seg_time, 1), float32,
        with cuts applied. Uses a memmap so RAM stays low, but the file is DENSE and
        uncompressed: each cut of all N events is ~N * 8.85 MB (~9.6 GB for all 1085
        events). Pass event_indices to export only a subset. Returns out_path.
        """
        events = list(range(len(self)) if event_indices is None else event_indices)
        per = self.segments_per_event(seg_wire, seg_time)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mm = open_memmap(out_path, mode="w+", dtype=np.float32,
                         shape=(len(events) * per, seg_wire, seg_time, 1))
        for k, e in enumerate(events):
            mm[k * per:(k + 1) * per] = self.segments(e, seg_wire, seg_time)
        mm.flush()
        return out_path


def export_all_cuts(h5_path, out_dir="inputData",
                    sizes=((864, 64), (64, 32), (18, 16)), event_indices=None):
    """Export one dense .npy per cut size from a processed plane-2 image .h5.

    Files are named ``<stem>_<wire>x<time>.npy`` in ``out_dir`` (stem taken from the .h5
    filename with a trailing '_images' removed), ready to feed training.py --input.
    Returns the list of written paths.
    """
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    if stem.endswith("_images"):
        stem = stem[: -len("_images")]
    written = []
    with WireImages(h5_path) as data:
        n = len(data)
        events = list(range(n) if event_indices is None else event_indices)
        per_event_mb = data.ds.shape[1] * data.ds.shape[2] * 4 / 1e6
        total_gb = len(sizes) * len(events) * per_event_mb / 1e3
        print("exporting %d cut(s) for %d/%d events -> ~%.2f GB total into %s/"
              % (len(sizes), len(events), n, total_gb, out_dir))
        for seg_wire, seg_time in sizes:
            path = os.path.join(out_dir, "%s_%dx%d.npy" % (stem, seg_wire, seg_time))
            data.export_npy(path, seg_wire, seg_time, event_indices=events)
            arr = np.load(path, mmap_mode="r")
            print("  wrote %s  shape=%s  (%.2f GB)"
                  % (path, arr.shape, os.path.getsize(path) / 1e9))
            written.append(path)
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export dense .npy segment files (one per cut: 864x64, 64x32, 18x16) "
                    "from a processed plane-2 image .h5, for training.py."
    )
    parser.add_argument("--h5", required=True, help="processed ..._plane2_images.h5 file")
    parser.add_argument("--out-dir", default="inputData", help="output directory (default: inputData)")
    parser.add_argument("--events", type=int, default=None,
                        help="number of events to export (default: ALL -- large; e.g. ~29 GB "
                             "for 3 cuts x 1085 events). Use e.g. --events 128 for a lean set.")
    args = parser.parse_args()
    ev = None if args.events is None else range(args.events)
    export_all_cuts(args.h5, out_dir=args.out_dir, event_indices=ev)
