"""Rosetta monomer stability features for the 77 in-house nanobodies.

For each nanobody monomer structure (chain B, from Boltz), normalize protonation-
state residue names, run FastRelax (ref2015), and record the relaxed total energy
(REU) and energy per residue as stability features. Lower (more negative) REU/res
~ more stable fold.

Parallelized with a process pool. Output: data/tm_predictor/rosetta_features.csv
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROSE = "/home/yoava/software/rosetta/source/bin/rosetta_scripts.default.linuxgccrelease"
XML = ROOT / "scripts/relax_score_monomer.xml"
SCRATCH = ROOT / "data/tm_predictor/rosetta_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
RENAME = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "CYX": "CYS",
          "GLH": "GLU", "ASH": "ASP", "LYN": "LYS"}
SCORE_RE = re.compile(r"Weighted score of total_score\s+(-?\d+\.\d+)")


def clean_pdb(src: Path, dst: Path):
    out = []
    for line in src.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 20:
            resn = line[17:20]
            if resn in RENAME:
                line = line[:17] + RENAME[resn] + line[20:]
        out.append(line)
    dst.write_text("\n".join(out) + "\n")


def run_one(args):
    pair_id, pdb_rel = args
    pdb = ROOT / pdb_rel
    if not pdb.exists():
        return pair_id, None
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        cl = Path(td) / "in.pdb"
        clean_pdb(pdb, cl)
        cmd = [ROSE, "-parser:protocol", str(XML), "-s", str(cl),
               "-out:path:all", td, "-nstruct", "1", "-relax:default_repeats", "1",
               "-ignore_unrecognized_res", "-overwrite"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return pair_id, None
        m = SCORE_RE.search(r.stdout)
        return pair_id, (float(m.group(1)) if m else None)


def main():
    feats = pd.read_csv(ROOT / "data/pep_patch/features.csv")
    nb = feats[feats["role"] == "tm_nb"][["pair_id", "pdb", "n_res"]].dropna(subset=["pdb"])
    jobs = list(zip(nb["pair_id"], nb["pdb"]))
    print(f"Running Rosetta FastRelax on {len(jobs)} monomers...")
    results = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            pid, score = f.result()
            results[pid] = score
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} done")

    nb = nb.copy()
    nb["rosetta_total_reu"] = nb["pair_id"].map(results)
    nb["rosetta_reu_per_res"] = nb["rosetta_total_reu"] / nb["n_res"]
    out = ROOT / "data/tm_predictor/rosetta_features.csv"
    nb[["pair_id", "n_res", "rosetta_total_reu", "rosetta_reu_per_res"]].to_csv(out, index=False)
    ok = nb["rosetta_total_reu"].notna().sum()
    print(f"Done: {ok}/{len(nb)} scored -> {out}")
    print(nb["rosetta_reu_per_res"].describe())


if __name__ == "__main__":
    main()
