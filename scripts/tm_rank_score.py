"""Thermostability ranking scorer (ESM-C 600M pairwise ranker).

Pure-sequence tool: give a VHH sequence (optionally a known target) and get a
within-target stability score + percentile, to prioritize candidates before
synthesis. Trained on the 77 in-house labels; expected within-target ranking
quality ~0.44 (vs 0.21 zero-shot).

Usage:
  # (re)train the artifact from the in-house labels
  python tm_rank_score.py train

  # score new sequences (FASTA or one-per-line), optionally for a known target
  python tm_rank_score.py score -i candidates.fasta -o scored.csv [--target <target_id>]

Training uses the cached ESM-C 600M embeddings; scoring embeds inputs on the fly
(reuses scripts/02_embed_esmc.py), so run in an env with esm+accelerate (esmfold2).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/tm_predictor/rank_model"
ART.mkdir(parents=True, exist_ok=True)
PCA_K = 16


# ----------------------------------------------------------------- train
def train():
    z = np.load(ROOT / "data/tm_predictor/inhouse_esmc_600m_tagstripped.npz",
                allow_pickle=True)
    emb = {n: e for n, e in zip(z["nb_name"], z["emb"])}
    tm = pd.read_csv(ROOT / "data/pep_patch/tm_joined.csv")
    X = np.stack([emb[n] for n in tm["nb_name"]]).astype(float)
    y = tm["tm_C"].to_numpy(float)
    tg = tm["target_id"].to_numpy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = PCA(n_components=PCA_K).fit(Xs)
    Z = pca.transform(Xs)

    D, L = [], []
    for t in np.unique(tg):
        idx = np.where(tg == t)[0]
        for a, b in itertools.combinations(idx, 2):
            if y[a] == y[b]:
                continue
            d = Z[a] - Z[b]
            lab = 1 if y[a] > y[b] else 0
            D += [d, -d]; L += [lab, 1 - lab]
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(
        np.array(D), np.array(L))
    w = clf.coef_[0]
    train_score = Z @ w

    joblib.dump({"scaler": scaler, "pca": pca, "w": w, "pca_k": PCA_K}, ART / "ranker.joblib")
    # reference score distributions for percentiles
    ref = {"global_scores": train_score.tolist(),
           "per_target": {t: (train_score[tg == t]).tolist() for t in np.unique(tg)}}
    (ART / "reference_scores.json").write_text(json.dumps(ref))
    # correlation sanity on training (in-sample, not a validation number)
    from scipy.stats import spearmanr
    rho = float(spearmanr(y, train_score).statistic)
    print(f"Trained ESM-C 600M ranker on {len(y)} binders. In-sample global rho={rho:.3f}")
    print(f"Artifact -> {ART/'ranker.joblib'}")


# ----------------------------------------------------------------- embedding
def _embed(seqs):
    spec = importlib.util.spec_from_file_location(
        "embed_esmc", str(ROOT / "scripts/02_embed_esmc.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules["embed_esmc"] = mod
    spec.loader.exec_module(mod)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.load_esmc("esmc_600m", dev).eval()
    with torch.inference_mode():
        emap = mod.embed_sequences(model, seqs, dev, "score")
    return np.stack([emap[s] for s in seqs]).astype(float)


def _read_seqs(path):
    names, seqs = [], []
    cur, buf = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur is not None:
                names.append(cur); seqs.append("".join(buf))
            cur, buf = line[1:].split()[0], []
        elif cur is None:  # one-per-line mode
            names.append(f"seq{len(names)+1}"); seqs.append(line)
        else:
            buf.append(line)
    if cur is not None:
        names.append(cur); seqs.append("".join(buf))
    return names, seqs


# ----------------------------------------------------------------- score
def score(inp, out, target=None):
    art = joblib.load(ART / "ranker.joblib")
    ref = json.loads((ART / "reference_scores.json").read_text())
    names, seqs = _read_seqs(inp)
    X = _embed(seqs)
    Z = art["pca"].transform(art["scaler"].transform(X))
    s = Z @ art["w"]

    g = np.array(ref["global_scores"])
    pct_g = [float((g < v).mean() * 100) for v in s]
    rows = {"name": names, "sequence": seqs, "stability_score": s,
            "percentile_global": pct_g}
    if target is not None and target in ref["per_target"]:
        tt = np.array(ref["per_target"][target])
        rows["percentile_in_target"] = [float((tt < v).mean() * 100) for v in s]
        rows["target"] = target
    df = pd.DataFrame(rows).sort_values("stability_score", ascending=False)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    s = sub.add_parser("score")
    s.add_argument("-i", "--input", required=True, help="FASTA or one-seq-per-line")
    s.add_argument("-o", "--output", default="tm_rank_scored.csv")
    s.add_argument("--target", default=None, help="known target_id for in-target percentile")
    a = p.parse_args()
    if a.cmd == "train":
        train()
    else:
        score(a.input, a.output, a.target)


if __name__ == "__main__":
    main()
