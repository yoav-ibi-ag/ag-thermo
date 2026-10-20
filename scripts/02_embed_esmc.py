"""Embed all unique target and NB sequences with local ESMC-300m.

Saves mean-pooled sequence embeddings plus pair feature matrix:
  [e_target || e_nb || e_target - e_nb || e_target * e_nb]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def mean_pool(embeddings: torch.Tensor, seq_len: int) -> np.ndarray:
    """Mean-pool residue embeddings, dropping BOS/EOS if present."""
    # embeddings: [1, L_tok, D]
    x = embeddings[0].detach().float().cpu()
    if x.shape[0] >= seq_len + 2:
        x = x[1 : seq_len + 1]
    else:
        x = x[:seq_len]
    return x.mean(dim=0).numpy()


# Architecture + local weight filename for open ESMC checkpoints.
ESMC_CONFIGS: dict[str, dict] = {
    "esmc_300m": {
        "d_model": 960,
        "n_heads": 15,
        "n_layers": 30,
        "data_root_key": "esmc-300",
        "weight_name": "esmc_300m_2024_12_v0.pth",
        "hf_repo": "biohub/esmc-300m-2024-12",
    },
    "esmc_600m": {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "data_root_key": "esmc-600",
        "weight_name": "esmc_600m_2024_12_v0.pth",
        "hf_repo": "biohub/esmc-600m-2024-12",
    },
    "esmc_6b": {
        "d_model": 2560,
        "n_heads": 40,
        "n_layers": 80,
        "data_root_key": "esmc-6b",
        "weight_name": "esmc_6b_2024_12_v0.pth",
        "hf_repo": "biohub/esmc-6b-2024-12",
        "skip_data_root_download": True,
    },
}


def _snapshot_dirs(cfg: dict) -> list[Path]:
    repo = cfg["hf_repo"].replace("/", "--")
    hub = Path.home() / ".cache/huggingface/hub" / f"models--{repo}"
    if not hub.is_dir():
        return []
    return sorted(p for p in (hub / "snapshots").glob("*") if p.is_dir())


def _complete_safetensors_dir(snapshot: Path) -> Path | None:
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        shards = list(snapshot.glob("*.safetensors"))
        if shards and not any(".incomplete" in p.name for p in shards):
            return snapshot
        return None
    weight_map = json.loads(index.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    for name in shards:
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            return None
    return snapshot


def _resolve_esmc_weight(cfg: dict) -> Path:
    """Find local .pth or a complete safetensors snapshot, without hitting the network if possible."""
    env = os.environ.get("ESMC_WEIGHT_PATH")
    if env and Path(env).is_file():
        return Path(env)

    for snap in _snapshot_dirs(cfg):
        pth = snap / "data/weights" / cfg["weight_name"]
        if pth.is_file():
            return pth.resolve()
        hits = list(snap.glob(cfg["weight_name"]))
        if hits:
            return hits[0].resolve()
        st = _complete_safetensors_dir(snap)
        if st is not None:
            return st.resolve()

    hub = Path.home() / ".cache/huggingface/hub" / f"models--{cfg['hf_repo'].replace('/', '--')}"
    if hub.is_dir():
        hits = list(hub.glob(f"**/{cfg['weight_name']}"))
        complete = [p.resolve() for p in hits if p.is_file() and ".incomplete" not in p.name]
        if complete:
            return complete[0]

    if cfg.get("skip_data_root_download"):
        raise FileNotFoundError(
            f"ESMC weights not found for {cfg['hf_repo']} ({cfg['weight_name']}). "
            "HF cache is missing a complete .pth / safetensors snapshot."
        )

    from esm.utils.constants.esm3 import data_root

    root = data_root(cfg["data_root_key"])
    pth = root / "data/weights" / cfg["weight_name"]
    if pth.is_file():
        return pth
    st = _complete_safetensors_dir(root)
    if st is not None:
        return st
    return pth


def _load_state_dict(weight_path: Path) -> dict:
    if weight_path.is_dir():
        from safetensors.torch import load_file

        index = weight_path / "model.safetensors.index.json"
        if index.is_file():
            shard_names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            shard_names = [p.name for p in sorted(weight_path.glob("*.safetensors"))]
        state: dict = {}
        for name in shard_names:
            state.update(load_file(str(weight_path / name), device="cpu"))
        remap: dict = {}
        for k, v in state.items():
            if k.startswith("esmc."):
                k = k[len("esmc.") :]
            if k.startswith("lm_head."):
                k = "sequence_head." + k[len("lm_head.") :]
            remap[k] = v
        return remap
    if weight_path.is_file():
        # HF hub blobs lose the .pth suffix after Path.resolve().
        try:
            return torch.load(weight_path, map_location="cpu", weights_only=False)
        except Exception:
            from safetensors.torch import load_file

            state = load_file(str(weight_path), device="cpu")
            if any(k.startswith("esmc.") for k in state):
                state = {
                    (k[len("esmc.") :] if k.startswith("esmc.") else k): v
                    for k, v in state.items()
                }
            return state
    raise FileNotFoundError(f"ESMC weights not found: {weight_path}")


def load_esmc(model_name: str, device: torch.device):
    """Load local ESMC weights.

    Current huggingface_hub.load_torch_model() is incompatible with the
    EvolutionaryScale checkpoint layout, so we load the .pth directly
    (same pattern ESM3 open weights use in esm.pretrained).
    """
    if model_name not in ESMC_CONFIGS:
        raise ValueError(
            f"Unsupported model {model_name!r}; choose from {sorted(ESMC_CONFIGS)}"
        )
    cfg = ESMC_CONFIGS[model_name]
    from accelerate import init_empty_weights
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    weight_path = _resolve_esmc_weight(cfg)
    if not (weight_path.is_file() or weight_path.is_dir()):
        raise FileNotFoundError(f"ESMC weights not found: {weight_path}")
    print(f"Loading {model_name} from {weight_path}")
    with init_empty_weights():
        model = ESMC(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
    state_dict = _load_state_dict(weight_path)
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing:
        raise RuntimeError(
            f"Failed to load {model_name}: {len(missing)} missing keys e.g. {missing[:8]}"
        )
    if unexpected:
        print(f"  ignored unexpected keys: {len(unexpected)} e.g. {unexpected[:5]}")
    return model.to(device)


def load_esmc_300m(device: torch.device):
    """Back-compat wrapper."""
    return load_esmc("esmc_300m", device)


def embed_sequences(
    model,
    sequences: list[str],
    device: torch.device,
    batch_note: str,
) -> dict[str, np.ndarray]:
    from esm.sdk.api import ESMProtein, LogitsConfig

    config = LogitsConfig(sequence=True, return_embeddings=True)
    out: dict[str, np.ndarray] = {}
    for seq in tqdm(sequences, desc=batch_note):
        if seq in out:
            continue
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        # move tensors to device if encode left them on CPU
        if hasattr(protein_tensor, "to"):
            protein_tensor = protein_tensor.to(device)
        logits_out = model.logits(protein_tensor, config)
        emb = logits_out.embeddings
        out[seq] = mean_pool(emb, seq_len=len(seq))
    return out


def build_pair_features(
    pairs: pd.DataFrame,
    target_emb: dict[str, np.ndarray],
    nb_emb: dict[str, np.ndarray],
) -> np.ndarray:
    rows = []
    for _, row in pairs.iterrows():
        et = target_emb[row["target_seq"]]
        en = nb_emb[row["nb_seq"]]
        feat = np.concatenate([et, en, et - en, et * en]).astype(np.float32)
        rows.append(feat)
    return np.stack(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/pairs_clean.csv"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--model-name",
        default="esmc_300m",
        choices=sorted(ESMC_CONFIGS),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path("embeddings") if args.model_name == "esmc_300m" else Path(
            f"embeddings_{args.model_name}"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Force CPU path to stay off GPU even if CUDA is visible
    if device.type == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    pairs = pd.read_csv(args.pairs)
    target_seqs = sorted(pairs["target_seq"].unique())
    nb_seqs = sorted(pairs["nb_seq"].unique())
    print(f"Unique targets to embed: {len(target_seqs)}")
    print(f"Unique NBs to embed: {len(nb_seqs)}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    model = load_esmc(args.model_name, device)
    model.eval()

    with torch.inference_mode():
        target_emb = embed_sequences(model, target_seqs, device, "targets")
        nb_emb = embed_sequences(model, nb_seqs, device, "nanobodies")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_ids = pairs.drop_duplicates("target_seq")[["target_id", "target_seq"]]
    nb_ids = pairs.drop_duplicates("nb_seq")[["nb_name", "nb_seq"]]

    target_matrix = np.stack([target_emb[s] for s in target_ids["target_seq"]], axis=0)
    nb_matrix = np.stack([nb_emb[s] for s in nb_ids["nb_seq"]], axis=0)
    pair_x = build_pair_features(pairs, target_emb, nb_emb)

    np.save(args.out_dir / "target_embeddings.npy", target_matrix)
    np.save(args.out_dir / "nb_embeddings.npy", nb_matrix)
    np.save(args.out_dir / "pair_features.npy", pair_x)
    target_ids.to_csv(args.out_dir / "target_embedding_ids.csv", index=False)
    nb_ids.to_csv(args.out_dir / "nb_embedding_ids.csv", index=False)
    pairs[["pair_id", "target_id", "nb_name", "bind", "has_kd", "kd_nM"]].to_csv(
        args.out_dir / "pair_feature_ids.csv", index=False
    )

    meta = {
        "model": args.model_name,
        "device": str(device),
        "n_pairs": int(len(pairs)),
        "n_targets": int(len(target_seqs)),
        "n_nbs": int(len(nb_seqs)),
        "embedding_dim": int(next(iter(target_emb.values())).shape[0]),
        "pair_feature_dim": int(pair_x.shape[1]),
        "pooling": "mean over residues (drop BOS/EOS when present)",
        "pair_feature": "concat(e_t, e_n, e_t-e_n, e_t*e_n)",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved embeddings -> {args.out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
