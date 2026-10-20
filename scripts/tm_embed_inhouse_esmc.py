"""Embed the 77 in-house campaign nanobodies (tag-stripped VHH) with ESM-C 600M.

The cached embeddings_esmc_600m/ nb embeddings were built from the *tagged*
nb_seq (…WGQGTQVTVSSAAAYPYDVPDYGSHHHHHH). The public_tm embeddings are
tag-stripped VHH. To use the public set as a pretrain domain we need the
in-house sequences embedded the same (tag-stripped) way. This script reuses the
loader in 02_embed_esmc.py and caches the result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tm_predictor"
OUT.mkdir(parents=True, exist_ok=True)


def _load_embed_module():
    spec = importlib.util.spec_from_file_location(
        "embed_esmc", str(ROOT / "scripts/02_embed_esmc.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_esmc"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    model_name = "esmc_600m"
    cache = OUT / f"inhouse_{model_name}_tagstripped.npz"
    # tag-stripped VHH comes straight from the NanoMelt prediction table.
    preds = pd.read_csv(ROOT / "data/nanomelt_eval/predictions.csv")
    seqs = preds["vhh_seq"].tolist()
    names = preds["nb_name"].tolist()
    if cache.exists():
        print(f"cache exists: {cache}")
        return

    mod = _load_embed_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}; embedding {len(seqs)} tag-stripped VHH")
    model = mod.load_esmc(model_name, device).eval()
    with torch.inference_mode():
        emb_map = mod.embed_sequences(model, seqs, device, "inhouse-vhh")
    mat = np.stack([emb_map[s] for s in seqs], axis=0).astype(np.float32)
    np.savez(cache, emb=mat, nb_name=np.array(names), vhh_seq=np.array(seqs))
    print(f"saved {mat.shape} -> {cache}")


if __name__ == "__main__":
    main()
