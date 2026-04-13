import argparse
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_CKPT = "/home/vnedungadi/CropFM/experiments/pretrain/contrastive_run_2/model_checkpoints/last.ckpt"
DEFAULT_PT = "/home/vnedungadi/CropFM/experiments/pretrain/dual_decoder_run/model_checkpoints/last.pt"


def torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location="cpu")


def is_tensor_dict(d: dict) -> bool:
    return isinstance(d, dict) and all(torch.is_tensor(v) for v in d.values())


def extract_state_dict(obj) -> dict:
    """
    Try common checkpoint formats:
    - Lightning ckpt: {"state_dict": {...}}
    - Plain state dict: {"weight_name": tensor, ...}
    """
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if is_tensor_dict(obj):
            return obj
    raise ValueError("Could not extract a tensor state_dict from this file.")


def normalize_key(k: str) -> str:
    # Remove wrappers commonly introduced by Lightning/DataParallel/model nesting.
    changed = True
    while changed:
        changed = False
        for p in ("model.", "module.", "_orig_mod."):
            if k.startswith(p):
                k = k[len(p) :]
                changed = True
    return k


def normalized_state_dict(sd: dict) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = normalize_key(k)
        if nk in out and out[nk].shape != v.shape:
            print(f"[warn] normalized key collision with different shape: {nk}")
        out[nk] = v
    return out


def split_keys(keys: Iterable[str]) -> Tuple[set, set]:
    keys = set(keys)
    encoder_like = {
        k
        for k in keys
        if k.startswith("encoder.") or k.startswith("tokenizer.") or ".encoder." in k
    }
    return keys, encoder_like


def compare_state_dicts(name_a: str, sd_a: dict, name_b: str, sd_b: dict) -> None:
    raw_a = set(sd_a.keys())
    raw_b = set(sd_b.keys())

    print("\n=== RAW KEY STATS ===")
    print(f"{name_a}: {len(raw_a)} keys")
    print(f"{name_b}: {len(raw_b)} keys")
    print(f"raw overlap: {len(raw_a & raw_b)}")
    print(f"only in {name_a}: {len(raw_a - raw_b)}")
    print(f"only in {name_b}: {len(raw_b - raw_a)}")

    nsd_a = normalized_state_dict(sd_a)
    nsd_b = normalized_state_dict(sd_b)
    norm_a_all, norm_a_encoder = split_keys(nsd_a.keys())
    norm_b_all, norm_b_encoder = split_keys(nsd_b.keys())

    print("\n=== NORMALIZED KEY STATS (strips model./module./_orig_mod.) ===")
    print(f"{name_a}: {len(norm_a_all)} keys")
    print(f"{name_b}: {len(norm_b_all)} keys")
    print(f"normalized overlap: {len(norm_a_all & norm_b_all)}")
    print(f"only in {name_a}: {len(norm_a_all - norm_b_all)}")
    print(f"only in {name_b}: {len(norm_b_all - norm_a_all)}")

    print("\n=== ENCODER/TOKENIZER KEY STATS (normalized) ===")
    print(f"{name_a}: {len(norm_a_encoder)} encoder/tokenizer-like keys")
    print(f"{name_b}: {len(norm_b_encoder)} encoder/tokenizer-like keys")
    print(f"encoder/tokenizer overlap: {len(norm_a_encoder & norm_b_encoder)}")
    print(f"encoder/tokenizer only in {name_a}: {len(norm_a_encoder - norm_b_encoder)}")
    print(f"encoder/tokenizer only in {name_b}: {len(norm_b_encoder - norm_a_encoder)}")

    common = sorted(norm_a_all & norm_b_all)
    shape_mismatches = []
    value_mismatches = []
    for k in common:
        ta = nsd_a[k]
        tb = nsd_b[k]
        if ta.shape != tb.shape:
            shape_mismatches.append((k, tuple(ta.shape), tuple(tb.shape)))
            continue
        if ta.dtype != tb.dtype:
            value_mismatches.append((k, f"dtype differs: {ta.dtype} vs {tb.dtype}"))
            continue
        if not torch.equal(ta, tb):
            # Quick, informative numeric diff.
            max_abs = (ta.float() - tb.float()).abs().max().item()
            value_mismatches.append((k, f"max_abs_diff={max_abs:.6g}"))

    print("\n=== COMMON KEY CONTENT CHECK ===")
    print(f"common normalized keys checked: {len(common)}")
    print(f"shape mismatches: {len(shape_mismatches)}")
    print(f"value mismatches (same shape): {len(value_mismatches)}")

    preview_n = 25
    if shape_mismatches:
        print(f"\nTop {min(preview_n, len(shape_mismatches))} shape mismatches:")
        for k, sa, sb in shape_mismatches[:preview_n]:
            print(f"  {k}: {name_a}{sa} vs {name_b}{sb}")

    if value_mismatches:
        print(f"\nTop {min(preview_n, len(value_mismatches))} value mismatches:")
        for k, msg in value_mismatches[:preview_n]:
            print(f"  {k}: {msg}")

    only_a = sorted(norm_a_all - norm_b_all)
    only_b = sorted(norm_b_all - norm_a_all)
    if only_a:
        print(f"\nTop {min(preview_n, len(only_a))} keys only in {name_a}:")
        for k in only_a[:preview_n]:
            print(f"  {k}")
    if only_b:
        print(f"\nTop {min(preview_n, len(only_b))} keys only in {name_b}:")
        for k in only_b[:preview_n]:
            print(f"  {k}")


def main():
    parser = argparse.ArgumentParser(description="Quick checkpoint diff utility.")
    parser.add_argument("--a", default=DEFAULT_CKPT, help="Path to first checkpoint")
    parser.add_argument("--b", default=DEFAULT_PT, help="Path to second checkpoint")
    args = parser.parse_args()

    print(f"Loading A: {args.a}")
    obj_a = torch_load_any(args.a)
    print(f"Loading B: {args.b}")
    obj_b = torch_load_any(args.b)

    sd_a = extract_state_dict(obj_a)
    sd_b = extract_state_dict(obj_b)
    print(f"Extracted state dict A with {len(sd_a)} keys")
    print(f"Extracted state dict B with {len(sd_b)} keys")

    compare_state_dicts("A", sd_a, "B", sd_b)


if __name__ == "__main__":
    main()
