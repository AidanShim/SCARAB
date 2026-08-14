# SCARAB cluster truth-labelling: one row per cluster, one CSV per cluster geometry.
#
# WHY THREE FILES.  A cluster exists in three different frames and its observables are
# NOT the same in each, so a single table would be ambiguous:
#   * RAW      - the variable-size crop written by tile_cluster.py pass 1
#                (clusters_raw_<tag>.h5).  The whole interaction, nothing lost.
#   * PADDED p - the fixed (target_wire, target_time) frame the autoencoder is actually
#                trained on (clusters_padded_<p>_<tag>.h5), one per percentile in PCTS.
#                pad_to_target() center-CROPS anything bigger than the target, so a big
#                cluster genuinely loses charge here -- its n_points, bbox, linearity and
#                neutrino purity all differ from the raw ones.
# Every file carries the same columns and the same row order (row i == row i of
# clusters_raw AND of clusters_padded_<p>), so they join on `row` with no bookkeeping.
#
# WHERE THE NUMBERS COME FROM
#   geometry / charge : clusterImages/clusters_raw_<tag>.h5 (pixels + per-cluster meta).
#                       The padded frames are NOT re-read -- pad_to_target is a pure
#                       center-crop + center-pad with no resampling, so a padded frame's
#                       pixels are exactly the raw pixels inside a sub-rectangle of the
#                       crop.  We take that sub-rectangle instead (verified identical),
#                       which is both exact and ~4x less I/O.
#   truth             : data/ubopendata/bnb_WithWire_NN.h5 -- hit_table (charge + position),
#                       edep_table (which Geant4 track deposited in each hit, and what
#                       fraction of its energy), particle_table (g4_id -> PDG), event_table
#                       (the event's neutrino: PDG, energy, CC/NC, vertex).
#                       These samples are simulated neutrino overlaid on REAL beam-off
#                       cosmic data, so a hit with no edep row has no Geant4 parent and is
#                       cosmic by construction -- the same truth rule the analysis notebook
#                       uses, here applied per FRAME and charge-weighted.
#
# THE REQUESTED OBSERVABLES (all computed inside that row's own frame)
#   n_points      counts of above-threshold pixels                  size
#   total_charge  sum of ADC                                        energy scale
#   bbox_area     bbox_w * bbox_h of the above-threshold pixels     spatial extent
#   aspect_ratio  bbox_w / bbox_h                                   elongation
#   linearity     charge-weighted PCA 1st-component variance frac   straightness
#   n_pca_90      components needed for 90% of the variance         dimensionality
#   charge_conc   charge in the brightest 10% of pixels / total     localization, dE/dx
#   hit_density   n_points / bbox_area                              compactness
#   percent_nu    neutrino-attributable hit charge / hit charge     neutrino purity
# plus ~40 further columns (shape, charge shape, hit-level semantic composition, the
# event's neutrino, and how much the frame cut away).  See MBOONE_TruthLabels/README.md,
# which this script writes alongside the CSVs.
#
# NOTE ON n_pca_90.  These are 2-D images, so PCA has exactly two components and
# n_pca_90 is 1 (linearity >= 0.90, i.e. track-like) or 2 (everything else).  It is kept
# because it is the requested column; `linearity` is the continuous version of it.
#
# RUN (from the SCARAB repo root or from main_project/MBOONE, scarab env):
#   conda run --no-capture-output -n scarab python main_project/MBOONE/cluster_truth_labels.py
#   ...  --force        rebuild even if the CSVs already exist
#   ...  --limit 5000   first N clusters only (quick smoke test; writes to *_sample.csv)
#   ...  --self-test    architecture / maths check, no data needed
# Env overrides: TRUTH_TILE_TAG (18x16), TRUTH_PCTS (50,75), TRUTH_PIX_THRESH (0).

import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import pandas as pd

# ------------------------------- Configuration -------------------------------
TILE_TAG = os.environ.get("TRUTH_TILE_TAG", "18x16")
PCTS = [int(x) for x in os.environ.get("TRUTH_PCTS", "50,75").split(",") if x.strip()]

# A pixel counts as "above threshold" when its ADC is > PIX_THRESH.  The preprocessing
# already applied the sample's own cutoff (10 ADC) and saturation (100 ADC), so every
# stored pixel is either 0 or >= 10 -- 0 is therefore the natural threshold and raising
# it means "on top of the cutoff the images were built with".
PIX_THRESH = float(os.environ.get("TRUTH_PIX_THRESH", "0.0"))

TIME_FACTOR = 10        # hit tick -> image pixel (matches the preprocessing downsample)
TOP_FRAC = 0.10         # charge_conc: fraction of the brightest pixels summed
LIN_CUT = 0.90          # n_pca_90: "90% of variance"

BLOCK_PX = 1 << 25      # raw pixels per HDF5 read (~134 MB of float32)

# pynuml-style semantic classes, same ids/order as the notebook's CLASS_ORDER.
CLASS_NAMES = ("cosmic", "other", "muon", "pion", "proton")
PDG_CLASS = {13: 2, 211: 3, 2212: 4}          # |pdg| -> class id; any other g4 parent = 1


def _find(cands):
    return next((p for p in cands if p.exists()), None)


def resolve_paths():
    """Locate the cluster file, the raw sample dir and the output dir from any cwd."""
    cimg = _find([Path("main_project/MBOONE/clusterImages"), Path("clusterImages"),
                  Path("../clusterImages")])
    if cimg is None:
        raise FileNotFoundError("clusterImages/ not found -- run tile_cluster.py pass 1 first")
    raw_dir = _find([Path("data/ubopendata"), Path("../../data/ubopendata"),
                     Path("../../../data/ubopendata")])
    out = cimg.parent / "MBOONE_TruthLabels"
    out.mkdir(parents=True, exist_ok=True)
    return cimg, raw_dir, out


# ------------------------- frame geometry (pad_to_target) ---------------------
def crop_window(h, w, tw, tt):
    """Which part of an (h, w) crop survives pad_to_target(image, tw, tt).

    pad_to_target center-CROPS an oversized image then center-pads, so the pixels that
    reach the padded frame are exactly crop[s0:s0+h2, s1:s1+w2].  Returns
    (s0, h2, s1, w2) in crop-local coordinates; nothing is lost when h2 == h and w2 == w.
    """
    s0 = max(0, (h - tw) // 2)
    s1 = max(0, (w - tt) // 2)
    return s0, min(h, tw), s1, min(w, tt)


# ------------------------------ geometry features -----------------------------
GEOM_COLS = ("n_points", "total_charge", "bbox_w", "bbox_h", "bbox_area", "aspect_ratio",
             "linearity", "n_pca_90", "charge_conc", "hit_density", "fill_frac",
             "charge_mean", "charge_max", "charge_std", "n_saturated",
             "pca_len1", "pca_len2", "pca_angle_deg", "centroid_wire", "centroid_time")

_EMPTY_GEOM = (0, 0.0, 0, 0, 0, np.nan, np.nan, 0, np.nan, np.nan, 0.0,
               np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan, np.nan)


def geom_features(r, c, v, frame_area, wire0, time0, sat):
    """Shape + charge observables of one cluster, in GEOM_COLS order.

    r / c  : wire (axis 0) and time (axis 1) pixel indices of the above-threshold pixels,
             in RAW-crop coordinates -- every returned quantity is either translation
             invariant or shifted back to full-image coordinates via wire0 / time0, so
             the caller never has to re-map a cropped frame's pixels.
    v      : their ADC values.  frame_area: pixels in the frame (for fill_frac).
    An empty cluster (nothing survived the frame or the threshold) returns zeros/NaN.
    """
    n = int(r.size)
    if n == 0:
        return _EMPTY_GEOM
    v = v.astype(np.float64, copy=False)
    q = float(v.sum())

    bw = int(r.max() - r.min()) + 1        # tight bbox of the LIT pixels, not of the frame
    bh = int(c.max() - c.min()) + 1
    area = bw * bh

    # charge_conc: what share of the charge sits in the brightest TOP_FRAC of the pixels.
    k = max(1, int(np.ceil(TOP_FRAC * n)))
    top = q if k >= n else float(np.partition(v, n - k)[n - k:].sum())

    # Charge-weighted 2-D PCA.  Weighting by ADC (rather than counting pixels) makes the
    # principal axis follow the ionization, so a bright core does not get outvoted by a
    # halo of threshold-grazing pixels.
    wt = v / q if q > 0 else np.full(n, 1.0 / n)
    mr = float((wt * r).sum())
    mc = float((wt * c).sum())
    dr = r - mr
    dc = c - mc
    srr = float((wt * dr * dr).sum())
    scc = float((wt * dc * dc).sum())
    src = float((wt * dr * dc).sum())
    tr = srr + scc
    if tr > 0:
        disc = np.sqrt(max(0.25 * (srr - scc) ** 2 + src * src, 0.0))
        l1 = 0.5 * tr + disc
        l2 = max(0.5 * tr - disc, 0.0)
        lin = l1 / (l1 + l2)
        ex, ey = l1 - scc, src                       # eigenvector of the larger eigenvalue
        if ex == 0.0 and ey == 0.0:
            ex, ey = 1.0, 0.0
        ang = (np.degrees(np.arctan2(ey, ex)) + 90.0) % 180.0 - 90.0   # [-90, 90), from wire axis
    else:
        # All the charge sits on one pixel: zero spread, so the covariance has no
        # direction at all.  Call that maximally linear / 1-dimensional rather than NaN,
        # so single-pixel blips do not poison downstream cuts on `linearity`.
        l1 = l2 = 0.0
        lin = 1.0
        ang = np.nan

    return (n, q, bw, bh, area,
            bw / bh,
            lin,
            1 if lin >= LIN_CUT else 2,
            top / q if q > 0 else np.nan,
            n / area,
            n / frame_area,
            q / n, float(v.max()), float(v.std()),
            int((v >= sat).sum()),
            np.sqrt(l1), np.sqrt(l2), ang,
            wire0 + mr, time0 + mc)


# --------------------------- pass A: geometry, all frames ---------------------
def geometry_pass(cluster_path, variants, limit=None):
    """Stream clusters_raw_<tag>.h5 once and fill GEOM_COLS for every frame.

    variants: [(name, target_or_None), ...] -- None means "the raw crop itself".
    Returns (meta, per_variant) where per_variant[name] holds the GEOM_COLS matrix plus
    the frame bookkeeping columns (frame_w/h, was_cropped, frac_area_kept, frac_charge_kept).
    """
    f = h5py.File(str(cluster_path), "r")
    n_all = int(f.attrs["n_clusters"])
    n = n_all if limit is None else min(int(limit), n_all)
    sat = float(f.attrs.get("saturation", np.inf))

    meta = dict(
        offsets=f["image_offsets"][:n + 1].astype(np.int64),
        width=f["image_width_px"][:n].astype(np.int64),        # wire extent of the crop
        height=f["image_height_px"][:n].astype(np.int64),      # time extent of the crop
        cluster_id=f["cluster_id"][:n],
        n_tiles=f["n_tiles"][:n],
        file_id=f["source_file_id"][:n].astype(np.int64),
        event=f["source_event_index"][:n].astype(np.int64),
        bbp=f["bbox_pixel_coords"][:n].astype(np.int64),       # (wire_lo, wire_hi, time_lo, time_hi)
        bbt=f["bbox_tile_coords"][:n].astype(np.int64),
        total_charge_h5=f["total_charge"][:n].astype(np.float64),
        source_files=[s.decode() if isinstance(s, bytes) else s for s in f.attrs["source_files"]],
        n_clusters=n,
        attrs={k: f.attrs[k] for k in ("seg_wire", "seg_time", "cutoff", "saturation",
                                       "charge_min", "image_wire", "image_time")
               if k in f.attrs},
    )

    out = {}
    for name, tgt in variants:
        out[name] = dict(
            g=np.empty((n, len(GEOM_COLS)), np.float64),
            frame_w=np.empty(n, np.int32), frame_h=np.empty(n, np.int32),
            was_cropped=np.zeros(n, np.int8),
            frac_area_kept=np.ones(n, np.float64),
            frac_charge_kept=np.ones(n, np.float64),
        )

    off, W, H = meta["offsets"], meta["width"], meta["height"]
    t0 = time.time()
    i = 0
    while i < n:
        j = i + 1                                    # at least one cluster per block
        while j < n and (off[j + 1] - off[i]) <= BLOCK_PX:
            j += 1
        lo, hi = int(off[i]), int(off[j])
        buf = f["image_data"][lo:hi]

        for ci in range(i, j):
            h, w = int(W[ci]), int(H[ci])
            img = buf[int(off[ci]) - lo:int(off[ci + 1]) - lo].reshape(h, w)
            m = img > PIX_THRESH
            r, c = np.nonzero(m)
            v = img[m]
            q_raw = float(v.sum())
            wire0, time0 = int(meta["bbp"][ci, 0]), int(meta["bbp"][ci, 2])

            for name, tgt in variants:
                d = out[name]
                if tgt is None:                      # the raw crop: nothing is cut
                    d["g"][ci] = geom_features(r, c, v, h * w, wire0, time0, sat)
                    d["frame_w"][ci], d["frame_h"][ci] = h, w
                    continue
                tw, tt = tgt
                s0, h2, s1, w2 = crop_window(h, w, tw, tt)
                if h2 == h and w2 == w:
                    rr, cc, vv = r, c, v
                else:
                    sel = (r >= s0) & (r < s0 + h2) & (c >= s1) & (c < s1 + w2)
                    rr, cc, vv = r[sel], c[sel], v[sel]
                    d["was_cropped"][ci] = 1
                    d["frac_area_kept"][ci] = (h2 * w2) / float(h * w)
                    d["frac_charge_kept"][ci] = (float(vv.sum()) / q_raw) if q_raw > 0 else np.nan
                # rr/cc stay in RAW-CROP coordinates (we subset, never re-index), so the
                # origin is still the crop's -- adding s0/s1 here would shift twice.
                d["g"][ci] = geom_features(rr, cc, vv, tw * tt, wire0, time0, sat)
                d["frame_w"][ci], d["frame_h"][ci] = tw, tt
        i = j
        if (i // 20000) != ((i - (j - i)) // 20000) or i == n:
            print("    geometry %7d / %d  (%.0fs)" % (i, n, time.time() - t0), flush=True)
    f.close()
    return meta, out


# ------------------------------ pass B: truth ---------------------------------
TRUTH_COLS = ("n_hits", "hit_charge", "nu_charge", "percent_nu", "percent_nu_hits",
              "n_hits_cosmic", "n_hits_other", "n_hits_muon", "n_hits_pion",
              "n_hits_proton", "dominant_class_id", "vtx_in_frame")

EVENT_COLS = ("nu_pdg", "nu_energy_gev", "is_cc", "nu_vtx_wire", "nu_vtx_time")


def _seq_offsets(seq):
    """(id -> row) and the cumulative row offsets of a pynuml event_id.seq_cnt table."""
    return ({int(e): i for i, e in enumerate(seq[:, 0])},
            np.concatenate([[0], np.cumsum(seq[:, 1])]).astype(np.int64))


def _event_hit_truth(rf, e, hrow, h_off, plane, wire, tim, integ,
                     erow, e_off, ehit, eg4, efr, prow, p_off, pg4, ppdg):
    """Plane-2 hits of event `e`, sorted by wire, with their charge and truth class.

    Returns (wire, time_px, charge, nu_share, class_id) or None if the event has no
    plane-2 hits.  nu_share is the fraction of the hit's charge that Geant4 attributes to
    simulated (i.e. neutrino) energy; a hit with no edep row keeps share 0 and class
    "cosmic", which is exactly right for these beam-off overlay samples.
    """
    i = hrow.get(int(e))
    if i is None:
        return None
    a, b = int(h_off[i]), int(h_off[i + 1])
    loc = np.nonzero(plane[a:b] == 2)[0]
    if loc.size == 0:
        return None

    nloc = b - a
    share = np.zeros(nloc)
    dom = np.full(nloc, -1, np.int64)
    j = erow.get(int(e))
    if j is not None:
        s, t = int(e_off[j]), int(e_off[j + 1])
        ids, fr, g4 = ehit[s:t], efr[s:t], eg4[s:t]
        ok = (ids >= 0) & (ids < nloc)               # edep hit_id is EVENT-local
        ids, fr, g4 = ids[ok], fr[ok], g4[ok]
        if ids.size:
            np.add.at(share, ids, fr)                # several tracks can share one hit
            np.clip(share, 0.0, 1.0, out=share)
            order = np.lexsort((-fr, ids))           # per hit, largest energy fraction first
            keep = np.ones(ids.size, bool)
            keep[1:] = ids[order][1:] != ids[order][:-1]
            dom[ids[order][keep]] = g4[order][keep]

    cls = np.zeros(nloc, np.int8)                    # 0 = cosmic (no Geant4 parent)
    have = dom >= 0
    if have.any():
        cls[have] = 1                                # "other" until the PDG says otherwise
        k = prow.get(int(e))
        if k is not None:
            s, t = int(p_off[k]), int(p_off[k + 1])
            gid, gpdg = pg4[s:t], np.abs(ppdg[s:t])
            o = np.argsort(gid)
            gs, ps = gid[o], gpdg[o]
            pos = np.clip(np.searchsorted(gs, dom[have]), 0, max(gs.size - 1, 0))
            hit_pdg = np.where(gs[pos] == dom[have], ps[pos], 0) if gs.size else np.zeros(int(have.sum()), np.int64)
            sub = np.ones(hit_pdg.size, np.int8)
            for pdg, cid in PDG_CLASS.items():
                sub[hit_pdg == pdg] = cid
            cls[have] = sub

    w = wire[a + loc].astype(np.int64)
    t_px = np.floor(tim[a + loc] / TIME_FACTOR).astype(np.int64)
    q = np.abs(integ[a + loc]).astype(np.float64)
    o = np.argsort(w, kind="stable")                 # wire-sorted -> searchsorted per cluster
    return w[o], t_px[o], q[o], share[loc][o], cls[loc][o]


def _fill_truth(dst, ci, w, t, q, share, cls, wlo, whi, tlo, thi, vtx):
    """Accumulate one cluster's hit truth inside the rectangle [wlo,whi) x [tlo,thi)."""
    m = (w >= wlo) & (w < whi) & (t >= tlo) & (t < thi)
    nh = int(m.sum())
    dst["n_hits"][ci] = nh
    if nh == 0:
        dst["hit_charge"][ci] = 0.0
        dst["nu_charge"][ci] = 0.0
        dst["percent_nu"][ci] = np.nan
        dst["percent_nu_hits"][ci] = np.nan
        dst["dominant_class_id"][ci] = -1
        for name in CLASS_NAMES:
            dst["n_hits_" + name][ci] = 0
    else:
        qs, cs = q[m], cls[m]
        tot = float(qs.sum())
        qnu = float((qs * share[m]).sum())
        dst["hit_charge"][ci] = tot
        dst["nu_charge"][ci] = qnu
        dst["percent_nu"][ci] = 100.0 * qnu / tot if tot > 0 else np.nan
        counts = np.bincount(cs, minlength=len(CLASS_NAMES))
        for cid, name in enumerate(CLASS_NAMES):
            dst["n_hits_" + name][ci] = counts[cid]
        dst["percent_nu_hits"][ci] = 100.0 * (nh - counts[0]) / nh
        dst["dominant_class_id"][ci] = int(np.argmax(counts))
    if vtx is not None:
        vw, vt = vtx
        dst["vtx_in_frame"][ci] = int(wlo <= vw < whi and tlo <= vt < thi)


def event_hit_truth(raw_path, event_index):
    """One event's plane-2 hits with their truth class -- for event displays.

    Reads ONLY that event's slices (not the whole file's columns), and derives the class the
    exact same way the CSV builder does, so a plotted dot and the `n_hits_*` columns can never
    disagree.  Returns (wire_px, time_px, charge, class_name) in FULL-image pixel coordinates,
    or (None, None, None, None) if the event has no plane-2 hits.
    """
    with h5py.File(str(raw_path), "r") as f:
        hrow, h_off = _seq_offsets(f["hit_table/event_id.seq_cnt"][:])
        i = hrow.get(int(event_index))
        if i is None:
            return None, None, None, None
        a, b = int(h_off[i]), int(h_off[i + 1])
        plane = f["hit_table/local_plane"][a:b, 0]
        loc = np.nonzero(plane == 2)[0]
        if loc.size == 0:
            return None, None, None, None
        wire = f["hit_table/local_wire"][a:b, 0].astype(np.int64)
        tick = f["hit_table/local_time"][a:b, 0]
        q = np.abs(f["hit_table/integral"][a:b, 0].astype(np.float64))

        dom = np.full(b - a, -1, np.int64)          # per hit, the g4 track that deposited most
        erow, e_off = _seq_offsets(f["edep_table/event_id.seq_cnt"][:])
        j = erow.get(int(event_index))
        if j is not None:
            s, t = int(e_off[j]), int(e_off[j + 1])
            ids = f["edep_table/hit_id"][s:t, 0].astype(np.int64)
            fr = f["edep_table/energy_fraction"][s:t, 0].astype(np.float64)
            g4 = f["edep_table/g4_id"][s:t, 0].astype(np.int64)
            ok = (ids >= 0) & (ids < b - a)
            ids, fr, g4 = ids[ok], fr[ok], g4[ok]
            if ids.size:
                order = np.lexsort((-fr, ids))
                keep = np.ones(ids.size, bool)
                keep[1:] = ids[order][1:] != ids[order][:-1]
                dom[ids[order][keep]] = g4[order][keep]

        pdg_of = {}
        prow, p_off = _seq_offsets(f["particle_table/event_id.seq_cnt"][:])
        k = prow.get(int(event_index))
        if k is not None:
            s, t = int(p_off[k]), int(p_off[k + 1])
            pdg_of = dict(zip(f["particle_table/g4_id"][s:t, 0].astype(np.int64).tolist(),
                              f["particle_table/g4_pdg"][s:t, 0].astype(np.int64).tolist()))

    cls = np.array([CLASS_NAMES[0] if g < 0                       # no Geant4 parent -> cosmic
                    else CLASS_NAMES[PDG_CLASS.get(abs(int(pdg_of.get(int(g), 0))), 1)]
                    for g in dom[loc]], dtype=object)
    return (wire[loc], np.floor(tick[loc] / TIME_FACTOR).astype(np.int64), q[loc], cls)


def truth_pass(meta, variants, raw_dir):
    """Hit-level Geant4 truth for every cluster, in every frame, + the event's neutrino."""
    n = meta["n_clusters"]
    per = {}
    for name, _ in variants:
        d = {c: np.full(n, np.nan) for c in TRUTH_COLS}
        for c in ("n_hits", "dominant_class_id", "vtx_in_frame"):
            d[c] = np.full(n, -1, np.int32)
        for name_c in CLASS_NAMES:
            d["n_hits_" + name_c] = np.full(n, -1, np.int32)
        per[name] = d
    ev = {c: np.full(n, np.nan) for c in EVENT_COLS}

    if raw_dir is None:
        print("  !! data/ubopendata not found -- every truth column stays empty")
        return per, ev

    bbp, fid, evt = meta["bbp"], meta["file_id"], meta["event"]
    for k, sfile in enumerate(meta["source_files"]):
        sel = np.nonzero(fid == k)[0]
        if sel.size == 0:
            continue
        path = raw_dir / sfile.replace("_plane2_images", "")
        if not path.exists():
            print("  file %02d: %s missing -> %d clusters left empty" % (k, path.name, sel.size))
            continue
        t0 = time.time()
        with h5py.File(str(path), "r") as rf:
            hrow, h_off = _seq_offsets(rf["hit_table/event_id.seq_cnt"][:])
            plane = rf["hit_table/local_plane"][:, 0]
            wire = rf["hit_table/local_wire"][:, 0]
            tim = rf["hit_table/local_time"][:, 0]
            integ = rf["hit_table/integral"][:, 0]
            erow, e_off = _seq_offsets(rf["edep_table/event_id.seq_cnt"][:])
            ehit = rf["edep_table/hit_id"][:, 0].astype(np.int64)
            eg4 = rf["edep_table/g4_id"][:, 0].astype(np.int64)
            efr = rf["edep_table/energy_fraction"][:, 0].astype(np.float64)
            prow, p_off = _seq_offsets(rf["particle_table/event_id.seq_cnt"][:])
            pg4 = rf["particle_table/g4_id"][:, 0].astype(np.int64)
            ppdg = rf["particle_table/g4_pdg"][:, 0].astype(np.int64)
            vrow, _ = _seq_offsets(rf["event_table/event_id.seq_cnt"][:])
            nu_pdg = rf["event_table/nu_pdg"][:, 0]
            nu_en = rf["event_table/nu_energy"][:, 0]
            is_cc = rf["event_table/is_cc"][:, 0]
            vtx_w = rf["event_table/nu_vtx_wire_pos"][:, 2]        # plane-2 wire
            vtx_t = rf["event_table/nu_vtx_wire_time"][:, 0]       # ticks

            order = sel[np.argsort(evt[sel], kind="stable")]       # group this file by event
            evs = evt[order]
            cuts = np.concatenate([[0], np.nonzero(np.diff(evs))[0] + 1, [order.size]])
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                e = int(evs[s0])
                rows = order[s0:s1]

                vtx = None
                vi = vrow.get(e)
                if vi is not None:
                    vw = int(vtx_w[vi])
                    vt = int(np.floor(vtx_t[vi] / TIME_FACTOR))
                    vtx = (vw, vt)
                    ev["nu_pdg"][rows] = int(nu_pdg[vi])
                    ev["nu_energy_gev"][rows] = float(nu_en[vi])
                    ev["is_cc"][rows] = int(is_cc[vi])
                    ev["nu_vtx_wire"][rows] = vw
                    ev["nu_vtx_time"][rows] = vt

                got = _event_hit_truth(rf, e, hrow, h_off, plane, wire, tim, integ,
                                       erow, e_off, ehit, eg4, efr, prow, p_off, pg4, ppdg)
                if got is None:
                    continue
                w, t, q, share, cls = got
                for ci in rows:
                    wlo, whi, tlo, thi = bbp[ci]
                    a = int(np.searchsorted(w, wlo, "left"))       # narrow once, reuse per frame
                    b = int(np.searchsorted(w, whi, "left"))
                    ws, ts, qs, ss, cs = w[a:b], t[a:b], q[a:b], share[a:b], cls[a:b]
                    h, wd = whi - wlo, thi - tlo
                    for name, tgt in variants:
                        if tgt is None:
                            _fill_truth(per[name], ci, ws, ts, qs, ss, cs,
                                        wlo, whi, tlo, thi, vtx)
                            continue
                        s0c, h2, s1c, w2 = crop_window(h, wd, *tgt)
                        _fill_truth(per[name], ci, ws, ts, qs, ss, cs,
                                    wlo + s0c, wlo + s0c + h2,
                                    tlo + s1c, tlo + s1c + w2, vtx)
        print("    truth file %02d: %6d clusters  (%.0fs)" % (k, sel.size, time.time() - t0),
              flush=True)
    return per, ev


# ------------------------------- assembly / output ----------------------------
def build_frame(name, meta, geom, truth, ev, is_nu_tile):
    """One variant's full table, as a DataFrame with the columns in a readable order."""
    n = meta["n_clusters"]
    g = geom["g"]
    col = {c: g[:, i] for i, c in enumerate(GEOM_COLS)}

    df = pd.DataFrame({
        # ---- identity: joins back to clusters_raw / clusters_padded row-for-row ----
        "row": np.arange(n, dtype=np.int64),
        "cluster_id": meta["cluster_id"].astype(np.int64),
        "source_file_id": meta["file_id"],
        "source_event_index": meta["event"],
        "n_tiles": meta["n_tiles"].astype(np.int64),
        # ---- where the cluster lives in the full 3456x640 plane-2 image ----
        "bbox_wire_lo": meta["bbp"][:, 0], "bbox_wire_hi": meta["bbp"][:, 1],
        "bbox_time_lo": meta["bbp"][:, 2], "bbox_time_hi": meta["bbp"][:, 3],
        "tile_row_lo": meta["bbt"][:, 0], "tile_row_hi": meta["bbt"][:, 1],
        "tile_col_lo": meta["bbt"][:, 2], "tile_col_hi": meta["bbt"][:, 3],
        # ---- the frame these features were measured in, and what it cost ----
        "crop_w": meta["width"], "crop_h": meta["height"],
        "frame_w": geom["frame_w"], "frame_h": geom["frame_h"],
        "was_cropped": geom["was_cropped"],
        "frac_area_kept": geom["frac_area_kept"],
        "frac_charge_kept": geom["frac_charge_kept"],
        # ---- the nine requested observables ----
        "n_points": col["n_points"].astype(np.int64),
        "total_charge": col["total_charge"],
        "bbox_area": col["bbox_area"].astype(np.int64),
        "aspect_ratio": col["aspect_ratio"],
        "linearity": col["linearity"],
        "n_pca_90": col["n_pca_90"].astype(np.int64),
        "charge_conc": col["charge_conc"],
        "hit_density": col["hit_density"],
        "percent_nu": truth["percent_nu"],
        # ---- extra shape / charge ----
        "bbox_w": col["bbox_w"].astype(np.int64),
        "bbox_h": col["bbox_h"].astype(np.int64),
        "fill_frac": col["fill_frac"],
        "charge_mean": col["charge_mean"],
        "charge_max": col["charge_max"],
        "charge_std": col["charge_std"],
        "n_saturated": col["n_saturated"].astype(np.int64),
        "pca_len1": col["pca_len1"],
        "pca_len2": col["pca_len2"],
        "pca_angle_deg": col["pca_angle_deg"],
        "centroid_wire": col["centroid_wire"],
        "centroid_time": col["centroid_time"],
        # ---- hit-level Geant4 truth, measured in THIS frame ----
        "percent_nu_hits": truth["percent_nu_hits"],
        "n_hits": truth["n_hits"],
        "hit_charge": truth["hit_charge"],
        "nu_charge": truth["nu_charge"],
        "n_hits_cosmic": truth["n_hits_cosmic"],
        "n_hits_other": truth["n_hits_other"],
        "n_hits_muon": truth["n_hits_muon"],
        "n_hits_pion": truth["n_hits_pion"],
        "n_hits_proton": truth["n_hits_proton"],
        "dominant_class": np.where(truth["dominant_class_id"] >= 0,
                                   np.array(CLASS_NAMES + ("",))[truth["dominant_class_id"]], ""),
        "is_nu_tile": is_nu_tile,
        # ---- the event's neutrino (same for every cluster of an event) ----
        "nu_pdg": ev["nu_pdg"],
        "nu_energy_gev": ev["nu_energy_gev"],
        "is_cc": ev["is_cc"],
        "nu_vtx_wire": ev["nu_vtx_wire"],
        "nu_vtx_time": ev["nu_vtx_time"],
        "vtx_in_frame": truth["vtx_in_frame"],
    })
    # How far the cluster's charge centroid sits from the neutrino vertex, in image
    # pixels.  Wire and time pixels are NOT the same physical length, so treat this as a
    # convenient proximity ranking rather than a distance in cm.
    dw = col["centroid_wire"] - ev["nu_vtx_wire"]
    dt = col["centroid_time"] - ev["nu_vtx_time"]
    df["vtx_dwire"] = dw
    df["vtx_dtime"] = dt
    df["vtx_dist"] = np.sqrt(dw * dw + dt * dt)
    df.attrs["variant"] = name
    return df


README = """# MBOONE cluster truth labels

Generated by `main_project/MBOONE/cluster_truth_labels.py`.  One row per extracted
cluster; row `i` is row `i` of `clusterImages/clusters_raw_{tag}.h5` **and** of
`clusterImages/clusters_padded_<pct>_{tag}.h5`, so the three tables join on `row`.

| file | frame the observables were measured in |
|---|---|
{table}

Why three files: `pad_to_target()` center-**crops** any cluster bigger than the padding
target, so in the padded frames a large cluster really has fewer pixels, less charge, a
different bounding box and a different neutrino purity than it does raw.  Comparing a
column across the three files tells you exactly what each padding target throws away
(`was_cropped`, `frac_area_kept`, `frac_charge_kept` quantify it directly).

Settings for this build: above-threshold = ADC > {thresh} (the images already carry the
sample cutoff {cutoff} / saturation {sat} ADC), `charge_conc` top fraction {topfrac},
`n_pca_90` variance target {lincut}, hit tick -> pixel divisor {tf}.

## Columns

### Identity
| column | meaning |
|---|---|
| `row` | row index into the cluster HDF5 files |
| `cluster_id` | cluster number within its source event |
| `source_file_id` | index into the source-file list below |
| `source_event_index` | event id within that file |
| `n_tiles` | active {tag} tiles in the cluster |
| `bbox_wire_lo/hi`, `bbox_time_lo/hi` | the raw crop's rectangle in the full 3456x640 plane-2 image (half-open) |
| `tile_row_lo/hi`, `tile_col_lo/hi` | the same box in tile units (inclusive) |

### Frame
| column | meaning |
|---|---|
| `crop_w`, `crop_h` | size of the raw crop (wire, time) -- identical in all three files |
| `frame_w`, `frame_h` | size of THIS file's frame |
| `was_cropped` | 1 if the raw crop did not fit and was center-cropped |
| `frac_area_kept` | surviving area / raw crop area |
| `frac_charge_kept` | surviving ADC / raw crop ADC |

### The nine requested observables
| column | measurement | observable |
|---|---|---|
| `n_points` | above-threshold pixels in the frame | size |
| `total_charge` | sum of ADC in the frame | energy scale |
| `bbox_area` | `bbox_w * bbox_h` of the lit pixels | spatial extent |
| `aspect_ratio` | `bbox_w / bbox_h` (wire extent / time extent) | elongation |
| `linearity` | charge-weighted PCA 1st-component variance fraction | straightness |
| `n_pca_90` | components for 90% of the variance | dimensionality |
| `charge_conc` | charge in the brightest 10% of pixels / total charge | localization, dE/dx |
| `hit_density` | `n_points / bbox_area` | compactness |
| `percent_nu` | 100 x neutrino-attributable hit charge / hit charge in the frame | neutrino purity |

`n_pca_90` is 1 or 2: these are 2-D images, so PCA has two components and the column is
just `linearity >= 0.90`.  `linearity` is its continuous form.

### Extra shape and charge
`bbox_w`, `bbox_h` (the lit-pixel bounding box), `fill_frac` (`n_points` / frame area),
`charge_mean`, `charge_max`, `charge_std`, `n_saturated` (pixels at the {sat} ADC
ceiling), `pca_len1`/`pca_len2` (RMS extent along the principal axes, pixels),
`pca_angle_deg` (principal axis vs the wire axis, in [-90, 90)), and
`centroid_wire`/`centroid_time` (charge-weighted centroid in FULL-image pixel
coordinates, so it is comparable across frames).

Degenerate clusters: when all the charge sits on a single pixel the covariance has no
direction, and the row reports `linearity = 1`, `n_pca_90 = 1`, `pca_len* = 0`,
`pca_angle_deg = NaN`.  A frame that kept no pixels at all reports `n_points = 0` with
NaN shape columns.

### Hit-level Geant4 truth, measured in this frame
These samples are simulated neutrino overlaid on **real beam-off cosmic data**, so a
reconstructed hit with no `edep_table` row has no Geant4 parent and is cosmic by
construction.  Charge is `|hit_table/integral|`; a hit's neutrino share is the summed
`edep_table/energy_fraction` over its Geant4 rows, clipped to 1.

| column | meaning |
|---|---|
| `n_hits` | plane-2 reconstructed hits inside the frame |
| `hit_charge`, `nu_charge` | their total charge, and its neutrino-attributable part |
| `percent_nu_hits` | 100 x (hits with a Geant4 parent) / `n_hits` -- the count-weighted twin of `percent_nu` |
| `n_hits_cosmic/other/muon/pion/proton` | hits by the class of their dominant depositor (\\|PDG\\| 13 / 211 / 2212; no parent = cosmic) |
| `dominant_class` | the most common of those classes |
| `is_nu_tile` | the coarse boolean label used elsewhere in the notebook: the cluster's TILE bbox touches a tile holding a Geant4 hit.  Identical in all three files |

`percent_nu` is NaN when the frame contains no reconstructed hits (`n_hits = 0`), which
happens for clusters whose charge is all sub-hit-threshold wire signal.  Note it is a
**hit**-level quantity while `total_charge` is a **wire**-level one, so they are not two
views of the same sum.

### The event's neutrino
`nu_pdg`, `nu_energy_gev`, `is_cc` (1 = charged current), `nu_vtx_wire` / `nu_vtx_time`
(the true vertex in plane-2 pixel coordinates), `vtx_in_frame` (1 if that vertex falls
inside this frame), and `vtx_dwire` / `vtx_dtime` / `vtx_dist` (centroid minus vertex, in
image pixels).  Wire and time pixels are different physical lengths, so `vtx_dist` is a
proximity ranking, not a distance in cm.

## Source files
{files}
"""


def write_readme(out_dir, meta, files_written):
    tag = TILE_TAG
    table = "\n".join("| `%s` | %s |" % (p.name, desc) for p, desc in files_written)
    files = "\n".join("%d. `%s`" % (i, s) for i, s in enumerate(meta["source_files"]))
    a = meta["attrs"]
    (out_dir / "README.md").write_text(README.format(
        tag=tag, table=table, files=files,
        thresh=("%g" % PIX_THRESH), cutoff=("%g" % float(a.get("cutoff", 10))),
        sat=("%g" % float(a.get("saturation", 100))), topfrac=("%g" % TOP_FRAC),
        lincut=("%g" % LIN_CUT), tf=TIME_FACTOR), encoding="utf-8")


def main(argv):
    force = "--force" in argv
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])

    cimg, raw_dir, out_dir = resolve_paths()
    cluster_path = cimg / ("clusters_raw_%s.h5" % TILE_TAG)
    if not cluster_path.exists():
        raise FileNotFoundError("%s not found -- run tile_cluster.py pass 1 first" % cluster_path)

    # Frame list: the raw crop plus one entry per padded percentile, targets read from
    # the padded files themselves so this can never drift from what was actually built.
    variants = [("raw", None)]
    for pct in PCTS:
        p = cimg / ("clusters_padded_%d_%s.h5" % (pct, TILE_TAG))
        if not p.exists():
            print("!! %s missing -> skipping the %dth-percentile table" % (p.name, pct))
            continue
        with h5py.File(str(p), "r") as pf:
            variants.append(("padded_%d" % pct,
                             (int(pf.attrs["target_wire"]), int(pf.attrs["target_time"]))))

    suffix = "_sample" if limit else ""
    paths = {name: out_dir / ("cluster_truth_%s_%s%s.csv" % (name, TILE_TAG, suffix))
             for name, _ in variants}
    if not force and all(p.exists() for p in paths.values()):
        print("all %d CSVs already exist in %s (use --force to rebuild):" % (len(paths), out_dir))
        for p in paths.values():
            print("   %s  (%.1f MB)" % (p.name, p.stat().st_size / 1e6))
        return 0

    print("cluster file : %s" % cluster_path)
    print("raw samples  : %s" % raw_dir)
    print("output       : %s" % out_dir)
    print("frames       : %s" % ", ".join(
        "%s%s" % (n, "" if t is None else " %dx%d" % t) for n, t in variants))

    t0 = time.time()
    print("\n[1/3] geometry + charge from the raw cluster pixels ...")
    meta, geom = geometry_pass(cluster_path, variants, limit=limit)
    n = meta["n_clusters"]

    # Cross-check: the raw frame's summed pixels must reproduce the charge tile_cluster
    # stored, otherwise the threshold or the ragged-offset unpacking is wrong.
    d = np.abs(geom["raw"]["g"][:, GEOM_COLS.index("total_charge")] - meta["total_charge_h5"])
    rel = d / np.maximum(meta["total_charge_h5"], 1e-9)
    print("      charge cross-check vs clusters_raw metadata: max rel. diff %.2e" % rel.max())

    print("\n[2/3] Geant4 hit truth for every frame ...")
    truth, ev = truth_pass(meta, variants, raw_dir)

    print("\n[3/3] writing CSVs ...")
    lab = cimg / ("cluster_truth_labels_%s.npy" % TILE_TAG)
    if lab.exists():
        is_nu_tile = np.load(lab)[:n].astype(np.int8)
    else:
        print("      !! %s missing -> is_nu_tile left empty" % lab.name)
        is_nu_tile = np.full(n, -1, np.int8)

    written = []
    for name, tgt in variants:
        df = build_frame(name, meta, geom[name], truth[name], ev, is_nu_tile)
        p = paths[name]
        df.to_csv(p, index=False, float_format="%.6g")
        desc = ("raw variable-size crop (`clusters_raw_%s.h5`)" % TILE_TAG if tgt is None
                else "%dx%d padded / center-cropped frame (`clusters_padded_%s_%s.h5`)"
                     % (tgt[0], tgt[1], name.split("_")[1], TILE_TAG))
        written.append((p, desc))
        nz = df["n_hits"] > 0
        print("   %-42s %7d rows x %2d cols  %6.1f MB" % (p.name, len(df), df.shape[1],
                                                          p.stat().st_size / 1e6))
        print("      median n_points %6.0f | median linearity %.3f | %5.1f%% cropped | "
              "%.1f%% of rows with hits have percent_nu > 50"
              % (df["n_points"].median(), df["linearity"].median(),
                 100.0 * df["was_cropped"].mean(),
                 100.0 * float((df.loc[nz, "percent_nu"] > 50).mean()) if nz.any() else 0.0))

    write_readme(out_dir, meta, written)
    print("\nwrote README.md; total %.1f min" % ((time.time() - t0) / 60.0))
    return 0


# ---------------------------------- self-test ---------------------------------
def self_test():
    """Maths + frame-geometry checks that need no data."""
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    # crop_window must agree with tile_cluster.pad_to_target on both branches.
    def pad_to_target(image, tw, tt):
        h, w = image.shape
        if h > tw or w > tt:
            s0, s1 = max(0, (h - tw) // 2), max(0, (w - tt) // 2)
            image = image[s0:s0 + min(h, tw), s1:s1 + min(w, tt)]
            h, w = image.shape
        out = np.zeros((tw, tt), image.dtype)
        out[(tw - h) // 2:(tw - h) // 2 + h, (tt - w) // 2:(tt - w) // 2 + w] = image
        return out

    rng = np.random.default_rng(0)
    good = True
    for _ in range(200):
        h, w = int(rng.integers(1, 300)), int(rng.integers(1, 300))
        tw, tt = 144, 112
        img = (rng.random((h, w)) * 100).astype(np.float32)
        s0, h2, s1, w2 = crop_window(h, w, tw, tt)
        good &= np.isclose(pad_to_target(img, tw, tt).sum(), img[s0:s0 + h2, s1:s1 + w2].sum())
    chk(good, "crop_window reproduces pad_to_target's surviving pixels")

    # A perfectly straight, uniformly bright line is maximally linear and 1-D.
    r = np.arange(40)
    g = dict(zip(GEOM_COLS, geom_features(r, r.copy(), np.full(40, 50.0), 1600, 0, 0, 100.0)))
    chk(abs(g["linearity"] - 1.0) < 1e-9, "diagonal line -> linearity 1")
    chk(g["n_pca_90"] == 1, "diagonal line -> n_pca_90 1")
    chk(abs(g["pca_angle_deg"] - 45.0) < 1e-6, "diagonal line -> 45 deg")
    chk(g["bbox_w"] == 40 and g["bbox_h"] == 40 and g["bbox_area"] == 1600, "line bbox")
    chk(abs(g["hit_density"] - 40 / 1600) < 1e-12, "line hit_density")

    # An isotropic blob is not linear; charge_conc of a flat blob is ~ the top fraction.
    rr, cc = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    g2 = dict(zip(GEOM_COLS, geom_features(rr.ravel(), cc.ravel(), np.full(400, 10.0),
                                           400, 0, 0, 100.0)))
    chk(abs(g2["linearity"] - 0.5) < 1e-9, "square blob -> linearity 0.5")
    chk(g2["n_pca_90"] == 2, "square blob -> n_pca_90 2")
    chk(abs(g2["charge_conc"] - 0.10) < 1e-9, "flat blob -> charge_conc == top fraction")
    chk(abs(g2["aspect_ratio"] - 1.0) < 1e-12, "square blob -> aspect_ratio 1")

    # One bright pixel dominating a faint halo must push charge_conc towards 1.
    v = np.full(50, 10.0)
    v[0] = 1000.0
    g3 = dict(zip(GEOM_COLS, geom_features(np.arange(50), np.zeros(50, int), v, 2500, 0, 0, 100.0)))
    chk(g3["charge_conc"] > 0.6, "one hot pixel -> charge_conc high (%.3f)" % g3["charge_conc"])
    chk(g3["n_saturated"] == 1, "n_saturated counts pixels at the ceiling")

    # Degenerate + empty rows must not produce garbage.
    g4 = dict(zip(GEOM_COLS, geom_features(np.array([5]), np.array([7]), np.array([30.0]),
                                           100, 3, 4, 100.0)))
    chk(g4["linearity"] == 1.0 and g4["n_pca_90"] == 1 and np.isnan(g4["pca_angle_deg"]),
        "single pixel -> linearity 1, n_pca_90 1, angle NaN")
    chk(g4["centroid_wire"] == 8 and g4["centroid_time"] == 11, "centroid shifts to full-image coords")
    g5 = dict(zip(GEOM_COLS, geom_features(np.array([], int), np.array([], int),
                                           np.array([]), 100, 0, 0, 100.0)))
    chk(g5["n_points"] == 0 and np.isnan(g5["linearity"]), "empty frame -> 0 points, NaN shape")

    print("\nself-test: %s" % ("OK" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
