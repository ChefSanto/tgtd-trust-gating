#!/usr/bin/env python3
"""Per-segment statistics for the pooled gate_log.csv(.gz).

The training script appends to one log across runs, so a new run starts wherever
`epoch` drops. This script streams the file in chunks, splits it into segments at
those drops, and for each segment reports:
  - rows, epochs covered
  - mean/std of chaos_score, for clean vs corrupted rows
  - AUROC of chaos_score for separating corrupted from clean (histogram estimate)
  - mean per-sample modality weights and the fraction of rows that are not the
    low template (weight_text > 0.40)
  - fraction of rows with chaos_score exactly 0

Usage:  python3 gate_log_segments.py /path/to/gate_log.csv.gz [out.csv]
Needs:  pandas, numpy
"""
import sys
import numpy as np
import pandas as pd

PATH = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "gate_log_segments.csv"
CHUNK = 2_000_000
NB = 200  # histogram bins over chaos in [0, 1]

head = pd.read_csv(PATH, nrows=5)
print("columns:", list(head.columns))
need = ["epoch", "chaos_score", "corruption_type",
        "weight_text", "weight_audio", "weight_visual"]
missing = [c for c in need if c not in head.columns]
if missing:
    sys.exit(f"missing columns {missing}; edit `need` to match the header above")


def new_seg(idx, first_epoch):
    return dict(seg=idx, first_epoch=first_epoch, last_epoch=first_epoch, rows=0,
                zero=0, wt=0.0, wa=0.0, wv=0.0, nonlow=0,
                hist_clean=np.zeros(NB), hist_corr=np.zeros(NB),
                s_clean=0.0, s2_clean=0.0, n_clean=0,
                s_corr=0.0, s2_corr=0.0, n_corr=0)


def finish(s):
    hc, hx = s["hist_clean"], s["hist_corr"]
    auc = np.nan
    if hc.sum() > 0 and hx.sum() > 0:
        cum_c = np.cumsum(hc) - hc  # clean strictly below each bin
        auc = (np.sum(hx * cum_c) + 0.5 * np.sum(hx * hc)) / (hc.sum() * hx.sum())
    n = max(s["rows"], 1)

    def ms(sm, sm2, k):
        if k == 0:
            return np.nan, np.nan
        m = sm / k
        return m, np.sqrt(max(sm2 / k - m * m, 0.0))

    mc, sc = ms(s["s_clean"], s["s2_clean"], s["n_clean"])
    mx, sx = ms(s["s_corr"], s["s2_corr"], s["n_corr"])
    return dict(segment=s["seg"], first_epoch=s["first_epoch"], last_epoch=s["last_epoch"],
                rows=s["rows"], frac_chaos_zero=s["zero"] / n,
                chaos_clean_mean=mc, chaos_clean_std=sc,
                chaos_corrupt_mean=mx, chaos_corrupt_std=sx, auroc_corrupt_vs_clean=auc,
                w_text=s["wt"] / n, w_audio=s["wa"] / n, w_visual=s["wv"] / n,
                frac_text_weight_gt_0p40=s["nonlow"] / n)


segs, cur, prev_epoch = [], None, None
for chunk in pd.read_csv(PATH, usecols=need, chunksize=CHUNK):
    ep = chunk["epoch"].to_numpy()
    # boundaries where epoch decreases (new run), including across chunk edges
    drops = np.flatnonzero(np.diff(ep) < 0) + 1
    if prev_epoch is not None and ep[0] < prev_epoch:
        drops = np.concatenate([[0], drops])
    edges = [0] + list(drops) + [len(chunk)]
    edges = sorted(set(edges))
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        part = chunk.iloc[a:b]
        if cur is None or (a in drops or (a == 0 and prev_epoch is not None and ep[0] < prev_epoch)):
            if cur is not None:
                segs.append(finish(cur))
            cur = new_seg(len(segs), int(part["epoch"].iloc[0]))
        c = part["chaos_score"].to_numpy(dtype=float)
        clean = (part["corruption_type"].astype(str) == "clean").to_numpy()
        cur["rows"] += len(part)
        cur["last_epoch"] = int(part["epoch"].iloc[-1])
        cur["zero"] += int((c == 0).sum())
        cur["wt"] += float(part["weight_text"].sum())
        cur["wa"] += float(part["weight_audio"].sum())
        cur["wv"] += float(part["weight_visual"].sum())
        cur["nonlow"] += int((part["weight_text"] > 0.40).sum())
        bins = np.clip((c * NB).astype(int), 0, NB - 1)
        cur["hist_clean"] += np.bincount(bins[clean], minlength=NB)
        cur["hist_corr"] += np.bincount(bins[~clean], minlength=NB)
        for tag, m in (("clean", clean), ("corr", ~clean)):
            cur["s_" + tag] += float(c[m].sum())
            cur["s2_" + tag] += float((c[m] ** 2).sum())
            cur["n_" + tag] += int(m.sum())
    prev_epoch = int(ep[-1])
if cur is not None:
    segs.append(finish(cur))

df = pd.DataFrame(segs)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(df.round(3).to_string(index=False))
print(f"\nsegments: {len(df)}  total rows: {int(df.rows.sum())}  -> wrote {OUT}")