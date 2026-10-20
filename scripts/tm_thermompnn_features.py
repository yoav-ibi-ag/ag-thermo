"""ThermoMPNN site-saturation stability features for the 77 in-house nanobodies.

ThermoMPNN predicts per-point-mutation ddG, not a per-protein stability scalar, so
for each monomer (chain B) we run the full single-mutation landscape (19 x L) and
aggregate it. The signal we are after is "stabilization headroom": a wild type that
is already near-optimal has few/weak stabilizing mutations available.

Convention check is done at runtime (Pro/Gly substitutions must come out
destabilizing); features are written in raw ThermoMPNN units either way.

Output: data/tm_predictor/thermompnn_features.csv
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TMPNN = Path("/home/yoava/software/ThermoMPNN")
sys.path.insert(0, str(TMPNN / "analysis"))
sys.path.insert(0, str(TMPNN))

from protein_mpnn_utils import alt_parse_PDB  # noqa: E402
from datasets import Mutation  # noqa: E402
from thermompnn_benchmarking import get_trained_model, ALPHABET  # noqa: E402
from SSM import get_ssm_mutations  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from tm_rosetta_features import clean_pdb  # noqa: E402

MODEL_PATH = TMPNN / "models/thermoMPNN_default.pt"
CHAIN = "B"


def load_model(device):
    config = {
        "training": {"num_workers": 8, "learn_rate": 0.001, "epochs": 100, "lr_schedule": True},
        "model": {
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
            "lr_schedule": True,
        },
    }
    cfg = OmegaConf.load(TMPNN / "local.yaml")
    merged = OmegaConf.merge(cfg, OmegaConf.create(config))
    merged.platform.thermompnn_dir = str(TMPNN)
    model = get_trained_model(model_name=str(MODEL_PATH), config=merged, override_custom=True)
    return model.eval().to(device)


def ssm_landscape(model, pdb_path, device):
    """Return a tidy frame of every point mutation's predicted ddG."""
    mut_pdb = alt_parse_PDB(str(pdb_path), CHAIN)
    muts = []
    for m in get_ssm_mutations(mut_pdb[0]):
        if m is None:
            muts.append(None)
            continue
        m = m.strip()
        wt, pos, mt = str(m[0]), int(str(m[1:-1])), str(m[-1])
        if wt not in ALPHABET or mt not in ALPHABET:
            muts.append(None)
            continue
        muts.append(Mutation(position=pos, wildtype=wt, mutation=mt, ddG=None,
                             pdb=mut_pdb[0]["name"]))
    with torch.no_grad():
        pred, _ = model(mut_pdb, muts)
    rows = [(m.position, m.wildtype, m.mutation, float(o["ddG"].cpu().item()))
            for m, o in zip(muts, pred) if m is not None]
    return pd.DataFrame(rows, columns=["position", "wildtype", "mutation", "ddg"])


def aggregate(df):
    """Collapse a mutation landscape into per-nanobody stability features.

    ThermoMPNN emits positive ddG = destabilizing (verified at runtime), so the
    per-site max is the worst available substitution at that position.
    """
    site_worst = df.groupby("position")["ddg"].max()
    return {
        "tmpnn_ddg_mean": df["ddg"].mean(),
        "tmpnn_ddg_std": df["ddg"].std(),
        "tmpnn_ddg_max": df["ddg"].max(),
        "tmpnn_ddg_min": df["ddg"].min(),
        "tmpnn_frac_destab": float((df["ddg"] > 0.5).mean()),
        "tmpnn_n_destab": int((df["ddg"] > 0.5).sum()),
        "tmpnn_site_worst_mean": site_worst.mean(),
        "tmpnn_site_worst_sum": site_worst.sum(),
        "tmpnn_n_sites": int(site_worst.size),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_model(device)

    feats = pd.read_csv(ROOT / "data/pep_patch/features.csv")
    nb = feats[feats["role"] == "tm_nb"][["pair_id", "pdb"]].dropna(subset=["pdb"])
    print(f"Running ThermoMPNN SSM on {len(nb)} monomers...")

    records, landscapes, failed = [], [], []
    with tempfile.TemporaryDirectory() as td:
        for i, (pair_id, pdb_rel) in enumerate(zip(nb["pair_id"], nb["pdb"]), 1):
            src = ROOT / pdb_rel
            if not src.exists():
                failed.append(pair_id)
                continue
            cl = Path(td) / "in.pdb"
            clean_pdb(src, cl)
            try:
                df = ssm_landscape(model, cl, device)
            except Exception as e:  # noqa: BLE001
                print(f"  {pair_id}: FAILED ({e})")
                failed.append(pair_id)
                continue
            records.append({"pair_id": pair_id, **aggregate(df)})
            df.insert(0, "pair_id", pair_id)
            landscapes.append(df)
            if i % 10 == 0:
                print(f"  {i}/{len(nb)} done", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(ROOT / "data/tm_predictor/thermompnn_features.csv", index=False)
    land = pd.concat(landscapes, ignore_index=True)
    land.to_csv(ROOT / "data/tm_predictor/thermompnn_landscape.csv.gz", index=False)

    # Sanity check on sign: Pro/Gly substitutions should be destabilizing.
    pg = land[land["mutation"].isin(["P", "G"])]["ddg"].mean()
    rest = land[~land["mutation"].isin(["P", "G"])]["ddg"].mean()
    print(f"\nsign check: mean ddg to Pro/Gly = {pg:.3f} vs all other = {rest:.3f}")
    print(f"  => {'positive = stabilizing' if pg < rest else 'positive = DEstabilizing'}")
    print(f"\nDone: {len(out)}/{len(nb)} scored ({len(failed)} failed)")
    print(out.describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
