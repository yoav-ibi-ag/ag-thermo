"""Embed the 77 in-house tag-stripped VHH with nanoBERT (NaturalAntibody/nanoBERT).

nanoBERT is a nanobody-specific RoBERTa MLM (hidden=320). Mean-pool last hidden
state over residues (drop BOS/EOS). Cache -> data/tm_predictor/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor"
# snapshot that carries config.json + tokenizer files
SNAP = Path("/home/yoava/.cache/huggingface/hub/models--NaturalAntibody--nanoBERT/"
            "snapshots/edc8182ad89a827f8737fa572c6b5fac6197e6b0")


def main():
    cache = OUT / "inhouse_nanobert.npz"
    if cache.exists():
        print("cache exists", cache); return
    preds = pd.read_csv(ROOT / "data/nanomelt_eval/predictions.csv")
    seqs, names = preds["vhh_seq"].tolist(), preds["nb_name"].tolist()

    tok = AutoTokenizer.from_pretrained(str(SNAP))
    model = AutoModel.from_pretrained(str(SNAP)).eval()
    embs = []
    with torch.inference_mode():
        for s in seqs:
            enc = tok(" ".join(list(s)), return_tensors="pt", truncation=True,
                      max_length=200)
            out = model(**enc).last_hidden_state[0]  # [L,320]
            embs.append(out[1:-1].mean(0).numpy())    # drop BOS/EOS
    mat = np.stack(embs).astype(np.float32)
    np.savez(cache, emb=mat, nb_name=np.array(names), vhh_seq=np.array(seqs))
    print("saved", mat.shape, "->", cache)


if __name__ == "__main__":
    main()
